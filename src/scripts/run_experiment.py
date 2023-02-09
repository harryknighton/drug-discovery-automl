from src.data import DatasetUsage
from src.models import build_uniform_gnn_architecture, GNNLayerType, PoolingFunction, ActivationFunction
from src.training import run_experiment

if __name__ == '__main__':
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

    run_experiment('mfpcba', 'AID1445', DatasetUsage.DROnly, architectures, DEFAULT_HYPER_PARAMETERS, [1424])
