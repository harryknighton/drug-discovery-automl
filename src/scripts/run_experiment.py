from src.models import ModelArchitecture, ActivationFunction, PoolingFunction, HyperParameters, \
    GNNArchitecture, RegressionLayerType, GNNLayerType
from src.training import run_experiment

if __name__ == '__main__':
    regression_layer = ModelArchitecture(
        layer_types=[RegressionLayerType.Linear],
        features=[256, 1],
        activation_funcs=[None],
        batch_normalise=[False]
    )
    gcn_architecture = GNNArchitecture(
        layer_types=[GNNLayerType.GCN, GNNLayerType.GCN, GNNLayerType.GCN],
        features=[113, 256, 256, 256],
        activation_funcs=[ActivationFunction.ReLU, ActivationFunction.ReLU, ActivationFunction.ReLU],
        pool_func=PoolingFunction.MEAN,
        batch_normalise=[True, True, False],
        regression_layer=regression_layer
    )

    params = HyperParameters(
        random_seed=1424,
        use_sd_readouts=False,
        k_folds=2,
        test_split=0.2,
        train_val_split=0.75,
        batch_size=32,
        early_stop_patience=30,
        early_stop_min_delta=0.01,
        lr=0.0001,
        max_epochs=10
    )
    run_experiment('pool_first', 'AID1445', [gcn_architecture], params, [1424])
