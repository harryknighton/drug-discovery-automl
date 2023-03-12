import gc
import pickle
from pathlib import Path

import hyperopt
import numpy as np
import pytorch_lightning as tl
import torch
from hyperopt import hp
from torch_geometric.data import LightningDataset, Dataset

from src.config import LOG_DIR, DEFAULT_LOGGER, DEFAULT_SAVE_TRIALS_EVERY
from src.data.scaling import Scaler
from src.data.utils import NamedLabelledDataset, BasicSplit, split_dataset
from src.models import PoolingFunction, GNNLayerType, ActivationFunction, GNNArchitecture, \
    build_uniform_regression_layer_architecture, BasicGNN
from src.reporting import generate_experiment_dir
from src.training import train_model, HyperParameters


def search_hyperparameters(
    dataset: NamedLabelledDataset,
    params: HyperParameters,
    search_space: dict,
    max_evals: int,
    experiment_name: str,
    precision: str = 'medium',
):
    torch.set_float32_matmul_precision(precision)
    tl.seed_everything(params.random_seeds[0], workers=True)

    # Load objects needed for HyperOpt
    experiment_dir = LOG_DIR / generate_experiment_dir(dataset.dataset, 'hyperopt_' + experiment_name)
    objective = _prepare_objective(dataset.dataset, dataset.label_scaler, params, experiment_dir)
    trials = _load_trials(experiment_dir)
    start = len(trials.trials)
    rstate = np.random.default_rng(params.random_seeds[0])

    DEFAULT_LOGGER.info(f"Running NAS experiment {experiment_name} at {experiment_dir}")

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
    DEFAULT_LOGGER.info(f"Best results: {trials.best_trial['result']['metrics']}")
    DEFAULT_LOGGER.info(f"Best architecture: {best_architecture}")


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


def _prepare_objective(dataset: Dataset, label_scaler: Scaler, params: HyperParameters, experiment_dir: Path):
    assert isinstance(params.dataset_split, BasicSplit)
    test_dataset, training_dataset = split_dataset(dataset, params.dataset_split.test_split)
    train_dataset, val_dataset = split_dataset(training_dataset, params.dataset_split.train_val_split)
    datamodule = LightningDataset(
        train_dataset, val_dataset, test_dataset,
        batch_size=params.batch_size, num_workers=params.num_workers
    )

    def objective(x):
        if not hasattr(objective, 'version'):
            objective.version = 0
        architecture = _convert_to_gnn_architecture(x, dataset.num_features, dataset.num_classes)
        model = BasicGNN(architecture)
        DEFAULT_LOGGER.debug("Evaluating architecture " + str(architecture))
        try:
            result = train_model(
                model=model,
                params=params,
                datamodule=datamodule,
                label_scaler=label_scaler,
                run_dir=experiment_dir,
                version=objective.version,
                save_logs=False,
                save_checkpoints=False,
            )
        except RuntimeError as e:
            DEFAULT_LOGGER.error(f"While training model error {e} was raised.")
            return {'status': hyperopt.STATUS_FAIL}
        cpu_result = {k: v.item() for k, v in result.items()}
        del result  # Free up memory
        objective.version += 1
        return {'loss': cpu_result['RootMeanSquaredError'], 'metrics': cpu_result, 'status': hyperopt.STATUS_OK}

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
        DEFAULT_LOGGER.info(f"Loading existing trials from {trials_path}")
        with open(trials_path, 'rb') as file:
            return pickle.load(file)
    else:
        DEFAULT_LOGGER.info("Creating new trial.")
        return hyperopt.Trials()


def _save_trials(trials: hyperopt.Trials, experiment_dir: Path) -> None:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    with open(experiment_dir / 'trials.pkl', 'wb') as out:
        pickle.dump(trials, out)
