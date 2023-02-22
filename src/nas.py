from pathlib import Path

import hyperopt
from hyperopt import hp
from torch_geometric.data import LightningDataset

from src.config import LOG_DIR, DEFAULT_N_FEATURES
from src.data import HTSDataset, split_dataset
from src.models import PoolingFunction, GNNLayerType, ActivationFunction, GNNArchitecture, \
    build_uniform_regression_layer_architecture
from src.parameters import DatasetUsage, HyperParameters, BasicSplit
from src.reporting import generate_experiment_dir
from src.training import train_model


def run_hyperopt(dataset_name: str, search_space: dict, params: HyperParameters, max_evals: int, experiment_name: str):
    name = 'hyperopt_' + experiment_name
    experiment_dir = LOG_DIR / generate_experiment_dir(dataset_name, params.dataset_usage, name)
    objective = _prepare_objective(dataset_name, params, experiment_dir)
    trials = hyperopt.Trials()
    best = hyperopt.fmin(
        fn=objective,
        space=search_space,
        algo=hyperopt.tpe.suggest,
        max_evals=max_evals,
        trials=trials
    )
    print(hyperopt.space_eval(search_space, best))


def _prepare_objective(dataset_name: str, params: HyperParameters, experiment_dir: Path):
    dataset = HTSDataset(dataset_name, DatasetUsage.DROnly)
    test_dataset, training_dataset = split_dataset(dataset, params.test_split)
    train_dataset, val_dataset = split_dataset(dataset, params.train_val_split)
    datamodule = LightningDataset(train_dataset, val_dataset, test_dataset, batch_size=params.batch_size,
                                  num_workers=params.num_workers)

    def objective(x):
        architecture = _convert_to_gnn_architecture(x)
        print(architecture)
        result = train_model(architecture, params, datamodule, dataset.scaler, experiment_dir)
        return {'loss': result['RootMeanSquaredError'], 'status': hyperopt.STATUS_OK}

    return objective


def _convert_to_gnn_architecture(space):
    layers = space['layers']
    regression_architecture = build_uniform_regression_layer_architecture(input_features=layers['hidden_features'])
    return GNNArchitecture(
        layer_types=layers['layer_types'],
        features=[DEFAULT_N_FEATURES] + [int(layers['hidden_features'])] * layers['num'],
        activation_funcs=layers['activation_funcs'],
        batch_normalise=[space['batch_normalise']] * layers['num'],
        pool_func=space['pool_func'],
        regression_layer=regression_architecture
    )


if __name__ == '__main__':
    params = HyperParameters(
        random_seed=0,
        dataset_usage=DatasetUsage.DROnly,
        dataset_split=BasicSplit(),
        test_split=0.1,
        train_val_split=0.9,
        batch_size=32,
        early_stop_patience=30,
        early_stop_min_delta=0,
        lr=3e-5,
        max_epochs=1,
        num_workers=0
    )

    simple_search_space = {
        'pool_func': hp.choice('pool_func', PoolingFunction),
        'batch_normalise': True,
        'layers': hp.choice('layers', [
            {
                'num': i,
                'layer_types': [hp.choice(f'type{i}{j}', [GNNLayerType.GCN, GNNLayerType.GAT]) for j in range(i)],
                'hidden_features': hp.quniform(f'features{i}', 16, 256, 8),
                'activation_funcs': [ActivationFunction.ReLU] * i
            }
            for i in range(1, 4)
        ]),
    }

    run_hyperopt('AID1445', simple_search_space, params, 10, 'test')
