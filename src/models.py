from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

from torch.nn import ReLU, Linear, Sequential
from torch_geometric.nn import (
    GCNConv, GINConv, GATConv, GATv2Conv,
    global_mean_pool, global_max_pool, global_add_pool,
    Sequential as SequentialGNN, BatchNorm
)


class LayerType(Enum):
    pass


class RegressionLayerType(LayerType):
    Linear = Linear


class GNNLayerType(LayerType):
    GCN = GCNConv
    GIN = GINConv
    GAT = GATConv
    GATv2 = GATv2Conv


class ActivationFunction(Enum):
    ReLU = ReLU


class PoolingFunction(Enum):
    MEAN = auto()
    MAX = auto()
    ADD = auto()

POOLING_FUNCTIONS = {
    PoolingFunction.MEAN: global_mean_pool,
    PoolingFunction.MAX: global_max_pool,
    PoolingFunction.ADD: global_add_pool,
}


@dataclass
class ModelArchitecture:
    layer_types: List[LayerType]
    features: List[int]
    activation_funcs: List[Optional[ActivationFunction]]
    batch_normalise: List[bool]

    def __str__(self):
        return f"ModelArchitecture({self._base_inner_str()})"

    def _base_inner_str(self):
        layer_names = [layer.name for layer in self.layer_types]
        activation_names = [func.name if func is not None else 'None' for func in self.activation_funcs]
        return (
            f"Layers: [{', '.join(layer_names)}], "
            f"Features: {self.features}, "
            f"Activation Functions: [{', '.join(activation_names)}]"
            f"Batch Normalise: {self.batch_normalise}"
        )

    def __post_init__(self):
        self._validate()

    def _validate(self):
        num_layers = len(self.layer_types)
        assert num_layers > 0
        assert len(self.features) == num_layers + 1
        assert len(self.activation_funcs) == num_layers
        assert len(self.batch_normalise) == num_layers


@dataclass
class GNNArchitecture(ModelArchitecture):
    layer_types: List[GNNLayerType]
    pool_func: PoolingFunction
    regression_layer: ModelArchitecture

    def __str__(self):
        return (
            "GNNArchitecture("
            f"{self._base_inner_str()}, "
            f"Pool Function: {self.pool_func.name}, "
            f"Regression Layer: {str(self.regression_layer)}"
            ")"
        )


def build_uniform_gnn_architecture(
    layer_type: GNNLayerType,
    num_layers: int,
    input_features: int,
    hidden_features: int,
    pool_func: PoolingFunction,
    batch_normalise: bool,
    activation: ActivationFunction,
    num_regression_layers: int,
    regression_layer_features: int,
) -> GNNArchitecture:
    regression_architecture = build_uniform_regression_layer_architecture(
        input_features=hidden_features,
        hidden_features=regression_layer_features,
        num_layers=num_regression_layers,
        batch_normalise=batch_normalise
    )
    return GNNArchitecture(
        layer_types=[layer_type] * num_layers,
        features=[input_features] + [hidden_features] * num_layers,
        activation_funcs=[activation] * num_layers,
        pool_func=pool_func,
        batch_normalise=[batch_normalise] * num_layers,
        regression_layer=regression_architecture
    )


def build_uniform_regression_layer_architecture(
    input_features: int,
    hidden_features: int = 128,
    num_layers: int = 3,
    batch_normalise: bool = True
) -> ModelArchitecture:
    return ModelArchitecture(
        layer_types=[RegressionLayerType.Linear] * num_layers,
        features=[input_features] + [hidden_features] * (max(num_layers - 1, 0)) + [1],
        activation_funcs=[ActivationFunction.ReLU] * (num_layers - 1) + [None],
        batch_normalise=[batch_normalise] * (num_layers - 1) + [False],
    )


def construct_gnn(arch: GNNArchitecture) -> SequentialGNN:
    global_inputs = "x, edge_index, batch"
    num_layers = len(arch.layer_types)
    layers = []
    for i in range(num_layers):
        layer_type = arch.layer_types[i]
        num_in = arch.features[i]
        num_out = arch.features[i + 1]
        activation = arch.activation_funcs[i]
        normalise = arch.batch_normalise[i]
        layer = _construct_layer(layer_type, num_in, num_out)
        layers.append((layer, "x, edge_index -> x"))
        if i == num_layers - 1:
            pool_func = POOLING_FUNCTIONS[arch.pool_func]
            layers.append((pool_func, "x, batch -> x"))
        if normalise:
            layers.append(BatchNorm(num_out))
        if activation is not None:
            layers.append(activation.value(inplace=True))

    return SequentialGNN(global_inputs, layers)


def construct_mlp(arch: ModelArchitecture) -> Sequential:
    layers = []
    for i in range(len(arch.layer_types)):
        layer_type = arch.layer_types[i]
        num_in = arch.features[i]
        num_out = arch.features[i + 1]
        activation = arch.activation_funcs[i]
        normalise = arch.batch_normalise[i]
        layer = _construct_layer(layer_type, num_in, num_out)
        layers.append(layer)
        if normalise:
            layers.append(BatchNorm(num_out))
        if activation is not None:
            layers.append(activation.value())

    return Sequential(*layers)


def _construct_layer(layer_type, num_in, num_out):
    if layer_type == GNNLayerType.GIN:
        # TODO: Add customisable layer architectures
        num_hidden = 2 * num_in
        mlp = Sequential(Linear(num_in, num_hidden), ReLU(), Linear(num_hidden, num_out))
        args = (mlp,)
    else:
        args = (num_in, num_out)
    return layer_type.value(*args)
