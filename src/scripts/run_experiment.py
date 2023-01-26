from src.models import ModelArchitecture, GNNLayer, ActivationFunction, PoolingFunction, HyperParameters
from src.training import run_experiment

if __name__ == '__main__':
    gcn_architecture = ModelArchitecture(
        name='gcn',
        layer_types=[GNNLayer.GCN, GNNLayer.GCN, GNNLayer.GCN],
        features=[133, 64, 16, 1],
        activation_funcs=[ActivationFunction.ReLU, ActivationFunction.ReLU, None],
        pool_func=PoolingFunction.MEAN
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
    run_experiment('no_fold', 'AID1445', [gcn_architecture], params, [1424])
