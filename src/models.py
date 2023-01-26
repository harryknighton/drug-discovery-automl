import inspect
import math
from dataclasses import dataclass
from enum import Enum
from typing import List

from torch.nn import ReLU, Linear, Sequential
from torch_geometric.nn import (
    GCNConv, GINConv, GATConv, GATv2Conv,
    global_mean_pool, global_max_pool, global_add_pool,
    Sequential as SequentialGNN
)


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


class GNNLayer(Enum):
    GCN = GCNConv
    GIN = GINConv
    GAT = GATConv
    GATv2 = GATv2Conv


class ActivationFunction(Enum):
    ReLU = ReLU


class PoolingFunction(Enum):
    MEAN = global_mean_pool
    MAX = global_max_pool
    ADD = global_add_pool


@dataclass
class ModelArchitecture:
    name: str  # For logging
    layer_types: List[GNNLayer]
    features: List[int]
    activation_funcs: List[ActivationFunction]
    pool_func: PoolingFunction

    def __str__(self):
        layer_names = [layer.name for layer in self.layer_types]
        activation_names = [func.name if func is not None else 'None' for func in self.activation_funcs]
        return inspect.cleandoc(f"""
            ModelArchitecture(
                Layers: [{', '.join(layer_names)}],
                Features: {self.features},
                Activation Functions: [{', '.join(activation_names)}],
                Pool Function: {self.pool_func.__name__}
            )
        """)


def construct_model(arch: ModelArchitecture) -> SequentialGNN:
    global_inputs = "x, edge_index, batch"
    layers = []
    for layer_type, num_in, num_out, activation in zip(
        arch.layer_types,
        arch.features[:-1],
        arch.features[1:],
        arch.activation_funcs
    ):
        layer = _construct_layer(layer_type, num_in, num_out)
        layers.append((layer, "x, edge_index -> x"))
        if activation is not None:
            layers.append(activation.value(inplace=True))
    layers.append((arch.pool_func, "x, batch -> x"))
    return SequentialGNN(global_inputs, layers)


def _construct_layer(layer_type, num_in, num_out):
    match layer_type:
        case GNNLayer.GIN:
            # TODO: Add customisable layer architectures
            num_hidden = int(math.sqrt(num_in + num_out))
            mlp = Sequential(Linear(num_in, num_hidden), ReLU(), Linear(16, num_hidden))
            args = (mlp,)
        case _: args = (num_in, num_out)
    return layer_type.value(*args)
