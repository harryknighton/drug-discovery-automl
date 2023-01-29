from src.models import ModelArchitecture, Layer, ActivationFunction, PoolingFunction, HyperParameters, \
    GNNArchitecture
from src.training import run_experiment

if __name__ == '__main__':
    regression_architecture = ModelArchitecture(
        layer_types=[Layer.Linear, Layer.Linear, Layer.Linear],
        features=[256, 128, 128, 1],
        activation_funcs=[ActivationFunction.ReLU, ActivationFunction.ReLU, None],
    )

    gcn_architecture = GNNArchitecture(
        layer_types=[Layer.GCN, Layer.GCN],
        features=[113, 256, 256],
        activation_funcs=[ActivationFunction.ReLU, None],
        pool_func=PoolingFunction.MEAN,
        regression_layer=regression_architecture
    )

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
        max_epochs=100
    )
    run_experiment('regression_layer', 'AID1445', [gcn_architecture], params, [1424])
