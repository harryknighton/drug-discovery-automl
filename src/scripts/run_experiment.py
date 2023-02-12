import argparse

from src.config import RANDOM_SEEDS
from src.data import DatasetUsage
from src.models import build_uniform_gnn_architecture, GNNLayerType, PoolingFunction, ActivationFunction, \
    HyperParameters
from src.training import run_experiment


def main():
    parser = argparse.ArgumentParser(description='Run Experiment')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--epochs', type=int, required=False, default=100)
    parser.add_argument('--use-mf-pcba-splits', action=argparse.BooleanOptionalAction, default=False)

    args = vars(parser.parse_args())
    _validate_args(args)

    params = HyperParameters(
        random_seed=1424,
        use_sd_readouts=False,
        k_folds=1,
        test_split=0.2,
        train_val_split=0.8,
        batch_size=32,
        early_stop_patience=30,
        early_stop_min_delta=0.01,
        lr=0.0001,
        max_epochs=args['epochs']
    )

    architectures = []
    for layer_type in GNNLayerType:
        architectures.append(
            build_uniform_gnn_architecture(
                layer_type=layer_type,
                num_layers=3,
                layer_width=256,
                pool_func=PoolingFunction.ADD,
                batch_normalise=True,
                activation=ActivationFunction.ReLU,
                num_regression_layers=2
            )
        )

    run_experiment(args['name'], args['dataset'], DatasetUsage.DROnly, architectures, params, [1424])


def _validate_args(args: dict):
    assert args['dataset'] in RANDOM_SEEDS.keys()
    assert args['epochs'] > 0


if __name__ == '__main__':
    main()
