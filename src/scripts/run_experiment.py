import argparse
import logging
import timeit
from pathlib import Path
from typing import Optional, Type

from src.config import LOG_DIR, DEFAULT_BATCH_SIZE, DEFAULT_LR, DEFAULT_EARLY_STOP_PATIENCE, \
    DEFAULT_EARLY_STOP_DELTA, DEFAULT_TEST_SPLIT, DEFAULT_TRAIN_VAL_SPLIT, DEFAULT_LABEL_SCALER, MF_PCBA_SEEDS
from src.data.hts import DatasetUsage
from src.data.scaling import fit_label_scaler, Scaler, StandardScaler, MinMaxScaler
from src.data.utils import get_dataset, NamedLabelledDataset, BasicSplit, MFPCBA, KFolds
from src.metrics import DEFAULT_METRICS
from src.models import build_uniform_gnn_architecture, GNNLayerType, PoolingFunction, ActivationFunction
from src.nas import search_hyperparameters, construct_search_space
from src.training import run_experiment, LitGNN, HyperParameters


def main():
    parser = argparse.ArgumentParser(description='Run Experiment')
    subparsers = parser.add_subparsers(required=True)

    experiment = subparsers.add_parser('experiment')
    experiment.set_defaults(func=_experiment)
    gnn_layers_strs = [x.name for x in GNNLayerType]
    pooling_strs = [x.name for x in PoolingFunction]
    data_usage_strs = [x.name for x in DatasetUsage]
    label_scalers = ['standard', 'minmax']
    experiment.add_argument('-N', '--name', type=str, required=True)
    experiment.add_argument('-D', '--dataset', type=str, required=True)
    experiment.add_argument('-d', '--dataset-usage', type=str, choices=data_usage_strs, default=None)
    experiment.add_argument('-L', '--label-scaler', type=str, choices=label_scalers, default=DEFAULT_LABEL_SCALER)
    experiment.add_argument('-e', '--epochs', type=int, default=100)
    experiment.add_argument('--use-mf-pcba-splits', action='store_true')
    experiment.add_argument('--k-folds', type=int)
    experiment.add_argument('--num-workers', type=int, default=0)
    experiment.add_argument('--precision', type=str, choices=['highest', 'high', 'medium'], default='highest')
    experiment.add_argument('-n', '--num-layers', type=int, nargs='+', default=[3])
    experiment.add_argument('-l', '--layer-types', type=str, nargs='+', choices=gnn_layers_strs, default=gnn_layers_strs)
    experiment.add_argument('-f', '--features', type=int, nargs='+', default=[128])
    experiment.add_argument('-r', '--num-regression-layers', type=int)
    experiment.add_argument('-w', '--regression-features', type=int)
    experiment.add_argument('-p', '--pooling-functions', type=str, nargs='+', choices=pooling_strs, default=pooling_strs)
    experiment.add_argument('-s', '--seeds', type=int, nargs='+', required=True)
    experiment.add_argument('--sd-ckpt', type=str, default=None)
    experiment.add_argument('--limit-batches', type=float, default=1.0)

    optimise = subparsers.add_parser('optimise')
    optimise.set_defaults(func=_optimise)
    optimise.add_argument('-N', '--name', type=str, required=True)
    optimise.add_argument('-D', '--dataset', type=str, required=True)
    optimise.add_argument('-d', '--dataset-usage', type=str, choices=data_usage_strs, default=None)
    optimise.add_argument('-L', '--label-scaler', type=str, choices=label_scalers, default=DEFAULT_LABEL_SCALER)
    optimise.add_argument('-S', '--search-space', type=str, choices=["simple"], required=True)
    optimise.add_argument('-e', '--max-evaluations', type=int, default=1000)
    optimise.add_argument('-s', '--seed', type=int, required=True)
    optimise.add_argument('--num-workers', type=int, default=0)
    optimise.add_argument('--precision', type=str, choices=['highest', 'high', 'medium'], default='highest')

    args = vars(parser.parse_args())
    args['func'](args)


def _experiment(args):
    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
    _validate_experiment_args(args)
    dataset_split = _resolve_dataset_split(args)
    dataset_usage = _resolve_dataset_usage(args)
    sd_ckpt_path = _resolve_sd_ckpt_path(args['sd_ckpt'], args['dataset'])
    layer_types = _resolve_layers(args)
    pool_funcs = _resolve_pooling_function(args)
    label_scaler_type = _resolve_label_scaler(args)

    raw_dataset = get_dataset(args['dataset'], dataset_usage=dataset_usage)
    label_scaler = fit_label_scaler(raw_dataset, label_scaler_type)
    if dataset_usage == DatasetUsage.DRWithSDReadouts:
        sd_model = LitGNN.load_from_checkpoint(sd_ckpt_path, label_scaler=label_scaler, metrics=DEFAULT_METRICS)
        raw_dataset.augment_dataset_with_sd_readouts(sd_model)
    dataset = NamedLabelledDataset(args['dataset'], raw_dataset, label_scaler)

    params = HyperParameters(
        random_seeds=args['seeds'],
        dataset_split=dataset_split,
        limit_batches=args['limit_batches'],
        batch_size=DEFAULT_BATCH_SIZE,
        early_stop_patience=DEFAULT_EARLY_STOP_PATIENCE,
        early_stop_min_delta=DEFAULT_EARLY_STOP_DELTA,
        lr=DEFAULT_LR,
        max_epochs=args['epochs'],
        num_workers=args['num_workers'],
        label_scaler=label_scaler_type,
    )

    architectures = []
    for layer_type in layer_types:
        for num_layers in args['num_layers']:
            for features in args['features']:
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
                            num_regression_layers=args['num_regression_layers'],
                            regression_layer_features=args['regression_features']
                        )
                    )

    start = timeit.default_timer()
    run_experiment(
        experiment_name=args['name'],
        dataset=dataset,
        label_scaler=label_scaler,
        architectures=architectures,
        params=params,
        precision=args['precision'],
    )
    end = timeit.default_timer()
    logging.info(f"Finished experiment in {end - start}s.")


def _optimise(args: dict):
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    search_space = construct_search_space(args['search_space'])
    dataset_usage = _resolve_dataset_usage(args)
    label_scaler_type = _resolve_label_scaler(args)

    raw_dataset = get_dataset(args['dataset'], dataset_usage=dataset_usage)
    label_scaler = fit_label_scaler(raw_dataset, label_scaler_type)
    dataset = NamedLabelledDataset(args['dataset'], raw_dataset, label_scaler)

    params = HyperParameters(
        random_seeds=[args['seed']],
        dataset_split=BasicSplit(test_split=DEFAULT_TEST_SPLIT, train_val_split=DEFAULT_TRAIN_VAL_SPLIT),
        label_scaler=label_scaler_type,
        batch_size=DEFAULT_BATCH_SIZE,
        early_stop_patience=DEFAULT_EARLY_STOP_PATIENCE,
        early_stop_min_delta=DEFAULT_EARLY_STOP_DELTA,
        lr=DEFAULT_LR,
        max_epochs=100,
        num_workers=args['num_workers'],
        limit_batches=1.0
    )

    search_hyperparameters(
        dataset=dataset,
        params=params,
        search_space=search_space,
        max_evals=args['max_evaluations'],
        experiment_name=args['name'],
        precision=args['precision']
    )


def _validate_experiment_args(args: dict):
    assert args['epochs'] > 0
    assert len(args['seeds']) > 0
    assert all(f > 0 for f in args['features'])
    assert all(n > 0 for n in args['num_layers'])
    assert args['num_regression_layers'] is None or args['num_regression_layers'] > 0
    assert args['regression_features'] is None or args['regression_features'] > 0
    assert args['limit_batches'] > 0
    if args['dataset'].startswith('AID'):
        if args['use_mf_pcba_splits']:
            assert args['dataset'] in MF_PCBA_SEEDS.keys()
            assert not args['k_folds']
        assert args['dataset_usage'] is not None
    else:
        assert not args['use_mf_pcba_splits']


def _resolve_dataset_split(args):
    if args['use_mf_pcba_splits']:
        dataset_split = MFPCBA(seeds=MF_PCBA_SEEDS[args['dataset']])
    elif args['k_folds']:
        dataset_split = KFolds(k=args['k_folds'], test_split=DEFAULT_TEST_SPLIT)
    else:
        dataset_split = BasicSplit(test_split=DEFAULT_TEST_SPLIT, train_val_split=DEFAULT_TRAIN_VAL_SPLIT)
    return dataset_split


def _resolve_dataset_usage(args) -> Optional[DatasetUsage]:
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


def _resolve_layers(args):
    return [GNNLayerType[name] for name in args['layer_types']]


def _resolve_pooling_function(args):
    return [PoolingFunction[name] for name in args['pooling_functions']]


def _resolve_label_scaler(args) -> Type[Scaler]:
    if args['label_scaler'] == 'standard':
        return StandardScaler
    elif args['label_scaler'] == 'minmax':
        return MinMaxScaler
    else:
        raise ValueError("Unknown label scaler " + str(args['label_scaler']))


if __name__ == '__main__':
    main()
