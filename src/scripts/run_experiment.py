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

    parser = argparse.ArgumentParser(description='Run Experiment')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--epochs', type=int, required=False, default=100)
    parser.add_argument('--use-mf-pcba-splits', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--k-folds', type=int, required=False)
    parser.add_argument('--num-workers', type=int, required=False, default=0)
    parser.add_argument('--precision', type=str, choices=['highest', 'high', 'medium'], default='highest')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])

    args = vars(parser.parse_args())
    _validate_args(args)

    if args['use_mf_pcba_splits']:
        dataset_split = MFPCBA(RANDOM_SEEDS[args['dataset']])
    elif args['k_folds']:
        dataset_split = KFolds(args['k_folds'])
    else:
        dataset_split = BasicSplit()

    params = HyperParameters(
        random_seed=args['seeds'],
        use_sd_readouts=False,
        dataset_split=dataset_split,
        test_split=0.2,
        train_val_split=0.8,
        batch_size=32,
        early_stop_patience=30,
        early_stop_min_delta=0.01,
        lr=0.0001,
        max_epochs=args['epochs'],
        num_workers=args['num_workers']
    )

    architectures = []
    for layer_type in GNNLayerType:
        architectures.append(
            build_uniform_gnn_architecture(
                layer_type=layer_type,
                num_layers=3,
                layer_width=256,
                pool_func=PoolingFunction.MEAN,
                batch_normalise=True,
                activation=ActivationFunction.ReLU,
                num_regression_layers=2
            )
        )

    start = timeit.default_timer()
    run_experiment(args['name'], args['dataset'], DatasetUsage.DROnly, architectures, params, args['seeds'], args['precision'])
    end = timeit.default_timer()
    logging.info(f"Finished experiment in {end - start}s.")


def _validate_args(args: dict):
    assert args['dataset'] in RANDOM_SEEDS.keys()
    assert args['epochs'] > 0
    assert not (args['use_mf_pcba_splits'] and args['k_folds'])
    assert len(args['seeds']) > 0


if __name__ == '__main__':
    main()
