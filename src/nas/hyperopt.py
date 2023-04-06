import gc
import pickle
from pathlib import Path
from typing import Callable, Dict, Any, Optional

import hyperopt
import numpy as np
import pytorch_lightning as tl
import torch
from hyperopt import hp, STATUS_OK, STATUS_FAIL
from torch_geometric.data import LightningDataset

from src.config import AUTOML_LOGGER, DEFAULT_SAVE_TRIALS_EVERY
from src.data.utils import NamedLabelledDataset, BasicSplit, split_dataset
from src.models import PoolingFunction, GNNLayerType, ActivationFunction, GNNArchitecture, \
    build_uniform_regression_layer_architecture, GNN
from src.nas.proxies import Proxy
from src.training import train_model, HyperParameters


def search_hyperparameters(
    experiment_dir: Path,
    dataset: NamedLabelledDataset,
    params: HyperParameters,
    search_space: dict,
    max_evals: int,
    proxy: Optional[Proxy] = None
):
    torch.set_float32_matmul_precision(params.precision)
    tl.seed_everything(params.random_seeds[0], workers=True)

    # Load objects needed for HyperOpt

    objective = _prepare_objective(dataset, params, experiment_dir, proxy)
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
            algo=hyperopt.tpe.suggest,
            max_evals=stop,
            trials=trials,
            rstate=rstate
        )
        _save_trials(trials, experiment_dir)
        with torch.no_grad():  # Prevent crashes
            gc.collect()
            torch.cuda.empty_cache()

    assert best is not None
    best_architecture = _convert_to_gnn_architecture(
        hyperopt.space_eval(search_space, best),
        input_features=dataset.dataset.num_features,
        output_features=dataset.dataset.num_classes,
    )
    AUTOML_LOGGER.info(f"Best results: {trials.best_trial['result']['metrics']}")
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
    proxy: Optional[Proxy] = None,
) -> Callable[[Any], Dict[str, Any]]:
    assert isinstance(params.dataset_split, BasicSplit)
    test_dataset, training_dataset = split_dataset(dataset.dataset, params.dataset_split.test_split)
    train_dataset, val_dataset = split_dataset(training_dataset, params.dataset_split.train_val_split)
    datamodule = LightningDataset(
        train_dataset, val_dataset, test_dataset,
        batch_size=params.batch_size, num_workers=params.num_workers
    )

    def objective(x):
        if not hasattr(objective, 'version'):
            objective.version = 0
        architecture = _convert_to_gnn_architecture(x, dataset.dataset.num_features, dataset.dataset.num_classes)
        model = GNN(architecture)
        AUTOML_LOGGER.debug("Evaluating architecture " + str(architecture))
        if proxy is not None:
            metric = proxy(model, dataset).item()
            loss = metric * -1. if proxy.higher_is_better else metric
            result = {'loss': loss, 'metrics': {proxy.__class__.__name__: metric}, 'status': STATUS_OK}
        else:
            try:
                metrics = train_model(
                    model=model,
                    params=params,
                    datamodule=datamodule,
                    label_scaler=dataset.label_scaler,
                    run_dir=experiment_dir,
                    version=objective.version,
                    save_logs=False,
                    save_checkpoints=False,
                )
                cpu_metrics = {k: v.item() for k, v in metrics.items()}
                result = {'loss': cpu_metrics['RootMeanSquaredError'], 'metrics': cpu_metrics, 'status': STATUS_OK}
            except RuntimeError as e:
                AUTOML_LOGGER.error(f"While training model error {e} was raised.")
                result = {'status': STATUS_FAIL}
        objective.version += 1
        return result

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
