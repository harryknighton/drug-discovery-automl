import gc
import pickle
import random
from pathlib import Path
from typing import Callable, Dict, Any, Optional

import hyperopt
import numpy as np
import pytorch_lightning as tl
import torch
from hyperopt import hp, STATUS_OK, STATUS_FAIL
from torch import Tensor
from torch_geometric.data import LightningDataset

from src.config import AUTOML_LOGGER, DEFAULT_SAVE_TRIALS_EVERY, LOSS_METRIC, EXPLAINABILITY_METRIC
from src.data.utils import NamedLabelledDataset, BasicSplit, split_dataset
from src.models import PoolingFunction, GNNLayerType, ActivationFunction, GNNArchitecture, \
    build_uniform_regression_layer_architecture, GNN
from src.nas.proxies import Proxy
from src.evaluation.reporting import save_run_results
from src.training import train_model, HyperParameters, perform_run
from src.types import Metrics


def search_hyperparameters(
    experiment_dir: Path,
    dataset: NamedLabelledDataset,
    params: HyperParameters,
    search_space: dict,
    algorithm: Callable,
    max_evals: int,
    noise_temperature: float,
    noise_decay: float,
    loss_explainability_ratio: float = 1.0,
    loss_proxy: Optional[Proxy] = None,
    explainability_proxy: Optional[Proxy] = None,
):
    torch.set_float32_matmul_precision(params.precision)

    if loss_proxy is not None:
        proxies, labels = get_fit_data(LOSS_METRIC, search_space, dataset, params, experiment_dir)
        loss_proxy.fit(proxies, labels, minimise_y=True)
    if explainability_proxy is not None:
        proxies, labels = get_fit_data(EXPLAINABILITY_METRIC, search_space, dataset, params, experiment_dir)
        explainability_proxy.fit(proxies, labels, minimise_y=False)

    # Load objects needed for HyperOpt
    objective = _prepare_objective(
        dataset=dataset,
        params=params,
        experiment_dir=experiment_dir,
        noise_temperature=noise_temperature,
        noise_decay=noise_decay,
        loss_explainability_ratio=loss_explainability_ratio,
        loss_proxy=loss_proxy,
        explainability_proxy=explainability_proxy
    )
    trials = _load_trials(experiment_dir)
    start = len(trials.trials)
    rstate = np.random.default_rng(params.random_seeds[0])

    if start >= max_evals:
        raise ValueError(f"max-evaluations should be greater than the number of trials performed so far ({start})")

    # Run NAS and save every `DEFAULT_SAVE_TRIALS_EVERY` evaluations
    best = None
    for interval in range(start, max_evals, DEFAULT_SAVE_TRIALS_EVERY):
        stop = min(interval + DEFAULT_SAVE_TRIALS_EVERY, max_evals)
        best = hyperopt.fmin(
            fn=objective,
            space=search_space,
            algo=algorithm,
            max_evals=stop,
            trials=trials,
            rstate=rstate
        )
        _save_trials(trials, experiment_dir)
        with torch.no_grad():  # Prevent crashes
            gc.collect()
            torch.cuda.empty_cache()

    if best is None:
        return
    best_hyperopt_architecture = hyperopt.space_eval(search_space, best)
    best_architecture = _convert_to_gnn_architecture(
        best_hyperopt_architecture,
        input_features=dataset.dataset.num_features,
        output_features=dataset.dataset.num_classes,
    )
    if loss_proxy is not None or explainability_proxy is not None:
        proxies, metrics = perform_run(dataset, best_architecture, params, experiment_dir, run_name='best_architecture')
    else:
        metrics = trials.best_trial['result']['metrics']
        save_run_results({str(best_architecture): metrics}, experiment_dir, 'best_architecture')
    AUTOML_LOGGER.info(f"Best results: {metrics}")
    AUTOML_LOGGER.info(f"Best architecture: {best_architecture}")


def construct_search_space(name: str):
    if name == 'simple':
        return {
            'pool_func': hp.choice('pool_func', PoolingFunction),
            'batch_normalise': True,
            'layers': hp.choice('layers', [
                {
                    'num': i,
                    'layer_types': [hp.choice(f'type{i}{j}', GNNLayerType) for j in range(i)],
                    'hidden_features': hp.quniform(f'features{i}', 16, 256, 8),
                    'activation_funcs': [ActivationFunction.ReLU] * i
                }
                for i in range(1, 4)
            ]),
        }


def _prepare_objective(
    dataset: NamedLabelledDataset,
    params: HyperParameters,
    experiment_dir: Path,
    noise_temperature: float,
    noise_decay: float,
    loss_explainability_ratio: float,
    loss_proxy: Optional[Proxy] = None,
    explainability_proxy: Optional[Proxy] = None
) -> Callable[[Any], Dict[str, Any]]:
    tl.seed_everything(params.random_seeds[0], workers=True)
    assert isinstance(params.dataset_split, BasicSplit)
    test_dataset, training_dataset = split_dataset(dataset.dataset, params.dataset_split.test_split)
    train_dataset, val_dataset = split_dataset(training_dataset, params.dataset_split.train_val_split)
    datamodule = LightningDataset(
        train_dataset, val_dataset, test_dataset,
        batch_size=params.batch_size, num_workers=params.num_workers
    )
    noise_generator = random.Random(params.random_seeds[0])

    def objective(x):
        architecture = _convert_to_gnn_architecture(x, dataset.dataset.num_features, dataset.dataset.num_classes)
        model = GNN(architecture)
        AUTOML_LOGGER.debug("Evaluating architecture " + str(architecture))
        metrics = {}
        status = STATUS_OK
        if loss_proxy is not None:
            metrics['loss_proxy'] = loss_proxy(model, dataset).item()
        if explainability_proxy is not None:
            metrics['explainability_proxy'] = explainability_proxy(model, dataset).item()
        if (
            loss_proxy is None and loss_explainability_ratio > 0. or
            explainability_proxy is None and loss_explainability_ratio < 1.
        ):
            try:
                gpu_metrics = train_model(
                    model=model,
                    params=params,
                    datamodule=datamodule,
                    label_scaler=dataset.label_scaler,
                    run_dir=experiment_dir,
                    version=objective.version,
                    save_logs=False,
                    save_checkpoints=False,
                )
                metrics.update({k: v.item() for k, v in gpu_metrics.items()})
            except RuntimeError as e:
                AUTOML_LOGGER.error(f"While training model error {e} was raised.")
                status = STATUS_FAIL

        result = _calculate_result(metrics, status)

        if objective.noise_temperature > 0 and 'loss' in result:
            noise = noise_generator.random()
            result['loss'] += noise * objective.noise_temperature
            objective.noise_temperature *= noise_decay

        objective.version += 1
        return result

    def _calculate_result(metrics: dict, status: str) -> dict:
        result = {'status': status}
        if not metrics:
            return result
        result['metrics'] = metrics
        test_loss = _calculate_loss('loss', metrics)
        explainability_loss = _calculate_loss('explainability', metrics)
        result['loss'] = loss_explainability_ratio * test_loss + (1 - loss_explainability_ratio) * explainability_loss
        return result

    def _calculate_loss(target: str, metrics: dict) -> float:
        match target:
            case 'loss': proxy = loss_proxy
            case 'explainability': proxy = explainability_proxy
            case _: raise ValueError("Unknown loss target")
        if proxy is not None:
            loss = metrics[target + '_proxy']
            if proxy.higher_is_better:
                loss *= -1
        elif target == 'loss' and loss_explainability_ratio > 0.0:
            loss = metrics[LOSS_METRIC]
        elif target == 'explainability' and loss_explainability_ratio < 1.0:
            loss = - metrics[EXPLAINABILITY_METRIC]
        else:
            loss = 0.
        return loss

    objective.version = 0
    objective.noise_temperature = noise_temperature

    return objective


def _convert_to_gnn_architecture(space: dict, input_features: int, output_features: int) -> GNNArchitecture:
    layers = space['layers']
    regression_architecture = build_uniform_regression_layer_architecture(
        input_features=int(layers['hidden_features']),
        output_features=output_features
    )
    return GNNArchitecture(
        layer_types=layers['layer_types'],
        features=[input_features] + [int(layers['hidden_features'])] * layers['num'],
        activation_funcs=layers['activation_funcs'],
        batch_normalise=[space['batch_normalise']] * layers['num'],
        pool_func=space['pool_func'],
        regression_layer=regression_architecture
    )


def _load_trials(experiment_dir: Path) -> hyperopt.Trials:
    trials_path = experiment_dir / 'trials.pkl'
    if trials_path.exists():
        AUTOML_LOGGER.info(f"Loading existing trials from {trials_path}")
        with open(trials_path, 'rb') as file:
            return pickle.load(file)
    else:
        AUTOML_LOGGER.info("Creating new trial.")
        return hyperopt.Trials()


def _save_trials(trials: hyperopt.Trials, experiment_dir: Path) -> None:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    with open(experiment_dir / 'trials.pkl', 'wb') as out:
        pickle.dump(trials, out)


def get_fit_data(
    target: str,
    search_space: dict,
    dataset: NamedLabelledDataset,
    params: HyperParameters,
    experiment_dir: Path,
    num_samples: int = 100,
) -> tuple[Metrics, Tensor]:
    samples_filepath = experiment_dir.parent / 'sampled_results.pt'
    if samples_filepath.exists():
        proxies, metrics = torch.load(samples_filepath)
    else:
        samples_filepath.parent.mkdir(parents=True, exist_ok=True)
        proxies, metrics = sample_proxies_metrics(search_space, num_samples, dataset, params)
        torch.save((proxies, metrics), samples_filepath)
    stacked_proxies = {proxy: torch.stack([sample[proxy] for sample in proxies]) for proxy in proxies[0]}
    labels = torch.stack([metric[target] for metric in metrics])
    return stacked_proxies, labels


def sample_proxies_metrics(
    search_space: dict, num_samples: int, dataset: NamedLabelledDataset, params: HyperParameters,
) -> tuple[list[Metrics], list[Metrics]]:
    proxies = []
    metrics = []
    input_features, output_features = dataset.dataset.num_features, dataset.dataset.num_classes
    architectures = _sample_architecture_space(search_space, num_samples, input_features, output_features)
    for architecture in architectures:
        model_proxies, model_metrics = perform_run(dataset, architecture, params, experiment_dir=None)
        proxies.append(list(model_proxies.values())[0])
        metrics.append(list(model_metrics.values())[0])
    return proxies, metrics


def _sample_architecture_space(
    search_space: dict, num_samples: int, in_features: int, out_features: int
) -> list[GNNArchitecture]:
    hyperopt_archs = [hyperopt.base.pyll.stochastic.sample(search_space) for _ in range(num_samples)]
    return [_convert_to_gnn_architecture(arch, in_features, out_features) for arch in hyperopt_archs]
