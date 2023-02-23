import argparse
import logging
import timeit

from src.config import RANDOM_SEEDS
from src.data import DatasetUsage, MFPCBA, KFolds, BasicSplit
from src.models import build_uniform_gnn_architecture, GNNLayerType, PoolingFunction, ActivationFunction
from src.parameters import HyperParameters
from src.training import run_experiment


def main():
    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)

    gnn_layers_strs = [x.name for x in GNNLayerType]
    pooling_strs = [x.name for x in PoolingFunction]

    parser = argparse.ArgumentParser(description='Run Experiment')
    parser.add_argument('-N', '--name', type=str, required=True)
    parser.add_argument('-D', '--dataset', type=str, required=True)
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('--use-mf-pcba-splits', action='store_true')
    parser.add_argument('--k-folds', type=int)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--precision', type=str, choices=['highest', 'high', 'medium'], default='highest')
    parser.add_argument('-n', '--num-layers', type=int, nargs='+', default=[3])
    parser.add_argument('-l', '--layer-types', type=str, nargs='+', choices=gnn_layers_strs, default=gnn_layers_strs)
    parser.add_argument('-f', '--features', type=int, nargs='+', default=[128])
    parser.add_argument('-r', '--num-regression-layers', type=int)
    parser.add_argument('-w', '--regression-features', type=int)
    parser.add_argument('-p', '--pooling-functions', type=str, nargs='+', choices=pooling_strs, default=pooling_strs)
    parser.add_argument('-s', '--seeds', type=int, nargs='+', required=True)

    args = vars(parser.parse_args())
    _validate_args(args)
    dataset_split = _resolve_dataset_split(args)
    layer_types = _resolve_layers(args)
    pool_funcs = _resolve_pooling_function(args)

    params = HyperParameters(
        random_seed=args['seeds'],
        dataset_usage=DatasetUsage.DROnly,
        dataset_split=dataset_split,
        test_split=0.1,
        train_val_split=0.9,
        batch_size=32,
        early_stop_patience=30,
        early_stop_min_delta=0,
        lr=3e-5,
        max_epochs=args['epochs'],
        num_workers=args['num_workers']
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
                            layer_width=features,
                            pool_func=pool_func,
                            batch_normalise=True,
                            activation=ActivationFunction.ReLU,
                            num_regression_layers=args['num_regression_layers'],
                            regression_layer_width=args['regression_features']
                        )
                    )

    start = timeit.default_timer()
    run_experiment(args['name'], args['dataset'], architectures, params, args['seeds'], args['precision'])
    end = timeit.default_timer()
    logging.info(f"Finished experiment in {end - start}s.")


def _validate_args(args: dict):
    assert args['dataset'] in RANDOM_SEEDS.keys()
    assert args['epochs'] > 0
    assert not (args['use_mf_pcba_splits'] and args['k_folds'])
    assert len(args['seeds']) > 0
    assert all(f > 0 for f in args['features'])
    assert all(n > 0 for n in args['num_layers'])
    assert args['num_regression_layers'] is None or args['num_regression_layers'] > 0
    assert args['regression_features'] is None or args['regression_features'] > 0


def _resolve_dataset_split(args):
    if args['use_mf_pcba_splits']:
        dataset_split = MFPCBA(RANDOM_SEEDS[args['dataset']])
    elif args['k_folds']:
        dataset_split = KFolds(args['k_folds'])
    else:
        dataset_split = BasicSplit()
    return dataset_split


def _resolve_layers(args):
    return [GNNLayerType[name] for name in args['layer_types']]


def _resolve_pooling_function(args):
    return [PoolingFunction[name] for name in args['pooling_functions']]


if __name__ == '__main__':
    main()
