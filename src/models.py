import math
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import List, Optional

from torch.nn import ReLU, Linear, Sequential
from torch_geometric.nn import (
    GCNConv, GINConv, GATConv, GATv2Conv,
    global_mean_pool, global_max_pool, global_add_pool,
    Sequential as SequentialGNN, BatchNorm
)

from src.config import DEFAULT_N_FEATURES


@dataclass
class HyperParameters:
    random_seed: int
    use_sd_readouts: bool
    k_folds: int
    test_split: float
    train_val_split: float
    batch_size: int
    early_stop_patience: int
    early_stop_min_delta: float
    lr: float
    max_epochs: int


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
    MEAN = partial(global_mean_pool)
    MAX = partial(global_max_pool)
    ADD = partial(global_add_pool)

    def __call__(self, *args, **kwargs):
        self.value(*args, **kwargs)


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
    layer_width: int,
    pool_func: PoolingFunction,
    batch_normalise: bool,
    activation: ActivationFunction,
    regression_layers: int,
) -> GNNArchitecture:
    return GNNArchitecture(
        layer_types=[layer_type] * num_layers,
        features=[DEFAULT_N_FEATURES] + [layer_width] * num_layers,
        activation_funcs=[activation] * num_layers,
        pool_func=pool_func,
        batch_normalise=[batch_normalise] * num_layers,
        regression_layer=build_regression_layer_architecture(layer_width, regression_layers)
    )


def build_regression_layer_architecture(input_features: int, layers: int) -> ModelArchitecture:
    return ModelArchitecture(
        layer_types=[RegressionLayerType.Linear] * layers,
        features=[input_features] * layers + [1],
        activation_funcs=[ActivationFunction.ReLU] * (layers - 1) + [None],
        batch_normalise=[False] * layers,
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
            layers.append((arch.pool_func.value, "x, batch -> x"))
        if normalise:
            layers.append(BatchNorm(num_out))
        if activation is not None:
            layers.append(activation.value(inplace=True))

    return SequentialGNN(global_inputs, layers)


def construct_mlp(arch: ModelArchitecture) -> Sequential:
    layers = []
    for layer_type, num_in, num_out, activation in zip(
        arch.layer_types,
        arch.features[:-1],
        arch.features[1:],
        arch.activation_funcs
    ):
        layer = _construct_layer(layer_type, num_in, num_out)
        layers.append(layer)
        if activation is not None:
            layers.append(activation.value())
    return Sequential(*layers)


def _construct_layer(layer_type, num_in, num_out):
    if layer_type == GNNLayerType.GIN:
        # TODO: Add customisable layer architectures
        num_hidden = int(math.sqrt(num_in + num_out))
        mlp = Sequential(Linear(num_in, num_hidden), ReLU(), Linear(num_hidden, num_out))
        args = (mlp,)
    else:
        args = (num_in, num_out)
    return layer_type.value(*args)
