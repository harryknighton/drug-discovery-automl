import argparse
import logging
import timeit
from pathlib import Path
from typing import Optional, Type, List

import tomli

from src.config import LOG_DIR, DEFAULT_BATCH_SIZE, DEFAULT_LR, DEFAULT_EARLY_STOP_PATIENCE, \
    DEFAULT_EARLY_STOP_DELTA, DEFAULT_TEST_SPLIT, DEFAULT_TRAIN_VAL_SPLIT, MF_PCBA_SEEDS, \
    EXPERIMENTS_DIR, DEFAULT_PRECISION
from src.data.hts import DatasetUsage
from src.data.scaling import fit_label_scaler, Scaler, StandardScaler, MinMaxScaler
from src.data.utils import get_dataset, NamedLabelledDataset, BasicSplit, MFPCBA, KFolds, DatasetSplit
from src.metrics import DEFAULT_METRICS
from src.models import build_uniform_gnn_architecture, GNNLayerType, PoolingFunction, ActivationFunction
from src.nas.hyperopt import search_hyperparameters, construct_search_space
from src.reporting import generate_experiment_dir
from src.training import run_experiment, LitGNN, HyperParameters


def main():
    parser = argparse.ArgumentParser(description='Run Experiment')
    parser.add_argument('-N', '--experiment-name', type=str, required=True)
    parser.add_argument('-D', '--dataset', type=str, required=True)
    parser.add_argument('-d', '--dataset-usage', type=str, required=True, choices=[d.name for d in DatasetUsage])
    args = vars(parser.parse_args())
    config = _load_config(args['experiment_name'])

    dataset_usage = _resolve_dataset_usage(args)
    raw_dataset = get_dataset(args['dataset'], dataset_usage=dataset_usage)
    label_scaler_type = _resolve_label_scaler(config)
    label_scaler = fit_label_scaler(raw_dataset, label_scaler_type)
    if dataset_usage == DatasetUsage.DRWithSDReadouts:
        sd_ckpt_path = _resolve_sd_ckpt_path(args['sd_ckpt'], args['dataset'])
        sd_model = LitGNN.load_from_checkpoint(sd_ckpt_path, label_scaler=label_scaler, metrics=DEFAULT_METRICS)
        raw_dataset.augment_dataset_with_sd_readouts(sd_model)
    dataset = NamedLabelledDataset(name=args['dataset'], dataset=raw_dataset, label_scaler=label_scaler)

    dataset_split = _resolve_dataset_split(config, args['dataset'])
    params = HyperParameters(
        random_seeds=config['seeds'],
        dataset_split=dataset_split,
        limit_batches=1.0,  # TODO: Remove limit batches
        batch_size=DEFAULT_BATCH_SIZE,
        early_stop_patience=DEFAULT_EARLY_STOP_PATIENCE,
        early_stop_min_delta=DEFAULT_EARLY_STOP_DELTA,
        lr=DEFAULT_LR,
        max_epochs=config['training']['max_epochs'],
        num_workers=config['data']['num_workers'],
        label_scaler=label_scaler_type,
        precision=DEFAULT_PRECISION
    )

    experiment_dir = LOG_DIR / generate_experiment_dir(dataset, args['experiment_name'])
    experiment_type = config['type']
    start = timeit.default_timer()
    if experiment_type == 'experiment':
        _experiment(experiment_dir, dataset, params, config['models'])
    elif experiment_type == 'nas':
        _nas(experiment_dir, dataset, params, config['search'])
    else:
        raise ValueError(f"Unknown experiment type {experiment_type}")
    end = timeit.default_timer()
    logging.info(f"Finished experiment in {end - start}s.")


def _experiment(experiment_dir: Path, dataset: NamedLabelledDataset, params: HyperParameters, model_config: dict):
    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
    layer_types = _resolve_layers(model_config)
    pool_funcs = _resolve_pooling_function(model_config)

    architectures = []
    for layer_type in layer_types:
        for num_layers in model_config['num_layers']:
            for features in model_config['features']:
                for pool_func in pool_funcs:
                    architectures.append(
                        build_uniform_gnn_architecture(
                            layer_type=layer_type,
                            num_layers=num_layers,
                            input_features=dataset.dataset.num_node_features,
                            output_features=dataset.dataset.num_classes,
                            hidden_features=features,
                            pool_func=pool_func,
                            batch_normalise=True,
                            activation=ActivationFunction.ReLU,
                            num_regression_layers=model_config['num_regression_layers'],
                            regression_layer_features=model_config['regression_features']
                        )
                    )

    run_experiment(
        experiment_dir=experiment_dir,
        dataset=dataset,
        architectures=architectures,
        params=params,
    )


def _nas(experiment_dir: Path, dataset: NamedLabelledDataset, params: HyperParameters, search_config: dict):
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    search_space = construct_search_space(search_config['search_space'])

    search_hyperparameters(
        experiment_dir=experiment_dir,
        dataset=dataset,
        params=params,
        search_space=search_space,
        max_evals=search_config['max_evaluations'],
    )


def _load_config(experiment_name: str):
    experiment_path = EXPERIMENTS_DIR / (experiment_name + '.toml')
    if not experiment_path.exists():
        pass
    with open(experiment_path, mode="rb") as fp:
        config = tomli.load(fp)
    _validate_config(config)
    return config


def _validate_config(config: dict) -> None:
    if config['training']['max_epochs'] <= 0:
        raise ValueError('Epochs must be greater than 0')
    if 'use_mf_pcba_splits' in config['data'] and 'k_folds' in config['data']:
        raise ValueError('Cannot specify both use_mf_pcba_splits and k_folds')
    if config['type'] == 'experiment':
        models = config['models']
        if len(config['seeds']) <= 0:
            raise ValueError('Must provide at least one seed')
        if any(f <= 0 for f in models['features']):
            raise ValueError('Layer features must be positive')
        if any(n <= 0 for n in models['num_layers']):
            raise ValueError('Number of layers must be positive')
        if models['num_regression_layers'] < 0:
            raise ValueError('Number of regression layers must be positive')
        if models['regression_features'] <= 0:
            raise ValueError('Regression layer features must be positive')
    elif config['type'] == 'nas':
        if not isinstance(config['seeds'], int):
            raise ValueError('Must only provide a single seed')
        if config['search']['max_evaluations'] <= 0:
            raise ValueError('Max evaluations must be positive')


def _resolve_dataset_split(config: dict, dataset_name: str) -> DatasetSplit:
    if config['data'].get('use_mf_pcba_splits'):
        if dataset_name.startswith('AID') and dataset_name not in MF_PCBA_SEEDS:
            raise RuntimeError(f'MF_PCBA seeds could not be found for dataset {dataset_name}')
        dataset_split = MFPCBA(seeds=MF_PCBA_SEEDS[dataset_name])
    elif config['data'].get('k_folds'):
        dataset_split = KFolds(k=config['data']['k_folds'], test_split=DEFAULT_TEST_SPLIT)
    else:
        dataset_split = BasicSplit(test_split=DEFAULT_TEST_SPLIT, train_val_split=DEFAULT_TRAIN_VAL_SPLIT)
    return dataset_split


def _resolve_dataset_usage(args: dict) -> Optional[DatasetUsage]:
    if args['dataset_usage'] is None:
        return None
    usage = DatasetUsage[args['dataset_usage']]
    if usage == DatasetUsage.DRWithSDReadouts and not args['sd_ckpt']:
        raise ValueError("If using SD readouts must specify sd-ckpt")
    return usage


def _resolve_sd_ckpt_path(sd_ckpt: str, dataset_name: str) -> Optional[Path]:
    if sd_ckpt is None:
        return None
    sd_ckpt_path = LOG_DIR / dataset_name / DatasetUsage.SDOnly.name / sd_ckpt
    if not sd_ckpt_path.exists():
        raise ValueError(f"No checkpoint at sd_ckpt {sd_ckpt_path}")
    if sd_ckpt_path.suffix != '.ckpt':
        raise ValueError(f"{sd_ckpt} is the wrong file type - should be .ckpt")
    return sd_ckpt_path


def _resolve_layers(model_config: dict) -> List[GNNLayerType]:
    return [GNNLayerType[name] for name in model_config['layer_types']]


def _resolve_pooling_function(model_config: dict) -> List[PoolingFunction]:
    return [PoolingFunction[name] for name in model_config['pooling_functions']]


def _resolve_label_scaler(config) -> Type[Scaler]:
    if config['data']['label_scaler'] == 'standard':
        return StandardScaler
    elif config['data']['label_scaler'] == 'minmax':
        return MinMaxScaler
    else:
        raise ValueError("Unknown label scaler " + str(config['data']['label_scaler']))


if __name__ == '__main__':
    main()
