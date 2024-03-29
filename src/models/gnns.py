"""Define GNN architectures and implement the models.

Copyright (c) 2023, Harry Knighton
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

from abc import ABC
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import List, Optional

import torch
from torch import Tensor
from torch.nn import ReLU, Linear, Sequential, ModuleList
from torch_geometric.nn import (
    GCNConv, GINConv, GATConv, GATv2Conv,
    global_mean_pool, global_max_pool, global_add_pool,
    BatchNorm
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
    # partial(f) needed to avoid f becoming member function
    MEAN = partial(global_mean_pool)
    MAX = partial(global_max_pool)
    ADD = partial(global_add_pool)


@dataclass
class ModelArchitecture(ABC):
    """Define a generic model architecture."""
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
            f"Activation Functions: [{', '.join(activation_names)}], "
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
class RegressionArchitecture(ModelArchitecture):
    """Define a conventional neural network architecture."""
    layer_types: List[RegressionLayerType]

    def __str__(self):
        return f"RegressionArchitecture({self._base_inner_str()})"


@dataclass
class GNNArchitecture(ModelArchitecture):
    """Define a GNN architecture."""
    layer_types: List[GNNLayerType]
    pool_func: PoolingFunction
    regression_layer: RegressionArchitecture

    def __str__(self):
        return (
            "GNNArchitecture("
            f"{self._base_inner_str()}, "
            f"Pool Function: {self.pool_func.name}, "
            f"Regression Layer: {str(self.regression_layer)}"
            ")"
        )


class GNNModule(torch.nn.Module, ABC):
    """Base class for all GNN implementations using `torch.nn.Module`"""
    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor):
        pass


class GraphBlock(GNNModule):
    """A single graph block layer in a GNN."""
    def __init__(
        self,
        layer_type: GNNLayerType,
        in_features: int,
        out_features: int,
        pooling: Optional[PoolingFunction],
        normalise: bool,
        activation: ActivationFunction
    ) -> None:
        super(GraphBlock, self).__init__()
        self.conv = _construct_layer(layer_type, in_features, out_features)
        self.pool = None if pooling is None else pooling.value
        self.normalise = None if normalise is None else BatchNorm(out_features, allow_single_element=True)
        self.activation = activation.value()

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        x = self.conv(x, edge_index)
        if self.pool is not None:
            x = self.pool(x, batch)
        if self.normalise is not None:
            x = self.normalise(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class RegressionLayer(torch.nn.Module):
    """Construct the model represented by a RegressionArchitecture."""
    def __init__(self, architecture: RegressionArchitecture):
        super(RegressionLayer, self).__init__()
        self.layers = ModuleList()
        for i in range(len(architecture.layer_types)):
            layer = _construct_layer(
                layer_type=architecture.layer_types[i],
                num_in=architecture.features[i],
                num_out=architecture.features[i+1]
            )
            self.layers.append(layer)
            if architecture.batch_normalise[i]:
                self.layers.append(BatchNorm(architecture.features[i+1], allow_single_element=True))
            if architecture.activation_funcs[i] is not None:
                self.layers.append(architecture.activation_funcs[i].value())

    def forward(self, x: Tensor):
        for layer in self.layers:
            x = layer(x)
        return x


class GNN(GNNModule):
    """Construct the model represented by a GNNArchitecture."""
    def __init__(self, architecture: GNNArchitecture):
        super(GNN, self).__init__()
        num_layers = len(architecture.layer_types)
        self.blocks = ModuleList([
            GraphBlock(
                layer_type=architecture.layer_types[i],
                in_features=architecture.features[i],
                out_features=architecture.features[i+1],
                pooling=architecture.pool_func if i == num_layers - 1 else None,
                normalise=architecture.batch_normalise[i],
                activation=architecture.activation_funcs[i]
            )
            for i in range(num_layers)
        ])
        self.regression_layer = RegressionLayer(architecture.regression_layer)

    def encode(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x, edge_index, batch)
        return x

    def readout(self, encodings: Tensor) -> Tensor:
        return self.regression_layer(encodings)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor):
        encodings = self.encode(x, edge_index, batch)
        return self.readout(encodings)


def _construct_layer(layer_type: LayerType, num_in: int, num_out: int) -> torch.nn.Module:
    if layer_type == GNNLayerType.GIN:
        num_hidden = 2 * num_in  # Fixed for convenience
        mlp = Sequential(Linear(num_in, num_hidden), ReLU(), Linear(num_hidden, num_out))
        args = (mlp,)
    else:
        args = (num_in, num_out)
    return layer_type.value(*args)


def build_uniform_gnn_architecture(
    layer_type: GNNLayerType,
    num_layers: int,
    input_features: int,
    output_features: int,
    hidden_features: int,
    pool_func: PoolingFunction,
    batch_normalise: bool,
    activation: ActivationFunction,
    num_regression_layers: int,
    regression_layer_features: int,
) -> GNNArchitecture:
    """Construct a GNN architecture with the same number of neurons in each layer."""
    regression_architecture = build_uniform_regression_layer_architecture(
        input_features=hidden_features,
        hidden_features=regression_layer_features,
        output_features=output_features,
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
    output_features: int,
    hidden_features: int = 128,
    num_layers: int = 3,
    batch_normalise: bool = True
) -> RegressionArchitecture:
    """Construct a conventional neural network architecture with the same number of neurons in each layer."""
    return RegressionArchitecture(
        layer_types=[RegressionLayerType.Linear] * num_layers,
        features=[input_features] + [hidden_features] * (max(num_layers - 1, 0)) + [output_features],
        activation_funcs=[ActivationFunction.ReLU] * (num_layers - 1) + [None],
        batch_normalise=[batch_normalise] * (num_layers - 1) + [False],
    )


def string_to_architecture(str_architecture: str) -> GNNArchitecture:
    """Construct a GNN architecture from its string representation

    Note: This was primarily used to reconstruct models from a previous experiment for which the checkpoint
    had been lost.
    """
    arch = str_architecture.replace(']Batch', '], Batch')  # For compatibility with an old buggy version
    arch = arch.replace(': ', '=')
    arch = arch.replace('Layers', 'layer_types')
    arch = arch.replace('Features', 'features')
    arch = arch.replace('Activation Functions', 'activation_funcs')
    arch = arch.replace('Batch Normalise', 'batch_normalise')
    arch = arch.replace('Pool Function', 'pool_func')
    arch = arch.replace('Regression Layer', 'regression_layer')
    for layer in GNNLayerType:
        arch = arch.replace(layer.name + ',', 'GNNLayerType.' + layer.name + ',')
        arch = arch.replace(layer.name + ']', 'GNNLayerType.' + layer.name + ']')
    for layer in RegressionLayerType:
        arch = arch.replace(layer.name, 'RegressionLayerType.' + layer.name)
    for pool_func in PoolingFunction:
        arch = arch.replace(pool_func.name, 'PoolingFunction.' + pool_func.name)
    for activation in ActivationFunction:
        arch = arch.replace(activation.name, 'ActivationFunction.' + activation.name)
    return eval(arch)
