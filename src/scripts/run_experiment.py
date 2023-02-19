import argparse
import logging
import timeit

from src.config import RANDOM_SEEDS
from src.data import DatasetUsage, MFPCBA, KFolds, BasicSplit
from src.models import build_uniform_gnn_architecture, GNNLayerType, PoolingFunction, ActivationFunction
from src.parameters import HyperParameters
from src.training import run_experiment


def main():
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)

    gnn_layers_strs = [x.name for x in GNNLayerType]

    parser = argparse.ArgumentParser(description='Run Experiment')
    parser.add_argument('-n', '--name', type=str, required=True)
    parser.add_argument('-d', '--dataset', type=str, required=True)
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('--use-mf-pcba-splits', action='store_true')
    parser.add_argument('--k-folds', type=int)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--precision', type=str, choices=['highest', 'high', 'medium'], default='highest')
    parser.add_argument('-l', '--layers', type=str, nargs='*', choices=gnn_layers_strs, default=gnn_layers_strs)
    parser.add_argument('-f', '--features', type=int, nargs='*', default=[128])
    parser.add_argument('-s', '--seeds', type=int, nargs='+', required=True)

    args = vars(parser.parse_args())
    _validate_args(args)
    dataset_split = _resolve_dataset_split(args)
    layers = _resolve_layers(args)

    params = HyperParameters(
        random_seed=args['seeds'],
        use_sd_readouts=False,
        dataset_split=dataset_split,
        test_split=0.2,
        train_val_split=0.75,
        batch_size=32,
        early_stop_patience=30,
        early_stop_min_delta=0.1,
        lr=0.0001,
        max_epochs=args['epochs'],
        num_workers=args['num_workers']
    )
    architectures = []
    for layer_type in layers:
        for features in args['features']:
            architectures.append(
                build_uniform_gnn_architecture(
                    layer_type=layer_type,
                    num_layers=3,
                    layer_width=features,
                    pool_func=PoolingFunction.MEAN,
                    batch_normalise=True,
                    activation=ActivationFunction.ReLU,
                    num_regression_layers=2
                )
            )

    start = timeit.default_timer()
    run_experiment(args['name'], args['dataset'], DatasetUsage.DROnly, [architectures[0]], params, args['seeds'], args['precision'])
    end = timeit.default_timer()
    logging.info(f"Finished experiment in {end - start}s.")


def _validate_args(args: dict):
    assert args['dataset'] in RANDOM_SEEDS.keys()
    assert args['epochs'] > 0
    assert not (args['use_mf_pcba_splits'] and args['k_folds'])
    assert len(args['seeds']) > 0


def _resolve_dataset_split(args):
    if args['use_mf_pcba_splits']:
        dataset_split = MFPCBA(RANDOM_SEEDS[args['dataset']])
    elif args['k_folds']:
        dataset_split = KFolds(args['k_folds'])
    else:
        dataset_split = BasicSplit()
    return dataset_split


def _resolve_layers(args):
    return [GNNLayerType[name] for name in args['layers']]


if __name__ == '__main__':
    main()
