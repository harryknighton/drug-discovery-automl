from src.data import DatasetUsage
from src.models import build_uniform_gnn_architecture, GNNLayerType, PoolingFunction, ActivationFunction, \
    HyperParameters
from src.training import run_experiment

if __name__ == '__main__':
    params = HyperParameters(
        random_seed=0,
        use_sd_readouts=False,
        k_folds=1,
        test_split=0.2,
        train_val_split=0.75,
        batch_size=32,
        early_stop_patience=30,
        early_stop_min_delta=0.01,
        lr=0.0001,
        max_epochs=5
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
                regression_layers=2
            )
        )

    run_experiment('uniform_test', 'AID1445', DatasetUsage.DROnly, architectures, params, [1424])
