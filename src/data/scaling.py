from abc import ABC
from typing import Type

import torch
from torch import Tensor
from torch_geometric.data import Dataset


class Scaler(torch.nn.Module, ABC):
    def fit(self, values: Tensor) -> Tensor:
        raise NotImplementedError()

    def transform(self, values: Tensor) -> Tensor:
        raise NotImplementedError()

    def inverse_transform(self, values: Tensor) -> Tensor:
        raise NotImplementedError()

    def fit_transform(self, values: Tensor) -> Tensor:
        self.fit(values)
        return self.transform(values)

    @staticmethod
    def _validate_input(values) -> None:
        if values.dim() != 2:
            raise ValueError("Values must be shape (n_samples, n_features) but got " + str(values.shape))


class StandardScaler(Scaler):
    def __init__(self, epsilon: float = 1e-7):
        """Adapted from https://gist.github.com/farahmand-m/8a416f33a27d73a149f92ce4708beb40"""
        super().__init__()
        self.register_buffer('means', Tensor(), persistent=False)
        self.register_buffer('stds', Tensor(), persistent=False)
        self.register_buffer('epsilon', torch.tensor([epsilon]), persistent=False)

    def fit(self, values: Tensor):
        self._validate_input(values)
        if values.shape[0] <= 1:
            raise ValueError("Must have more than one sample to fit to, got " + str(values.shape[0]))
        self.means = torch.mean(values, dim=0)
        self.stds = torch.std(values, dim=0)
        assert len(self.means) == len(self.stds) == values.shape[-1]

    def transform(self, values: Tensor):
        self._validate_input(values)
        return (values - self.means) / (self.stds + self.epsilon)

    def inverse_transform(self, values):
        self._validate_input(values)
        return values * (self.stds + self.epsilon) + self.means


class MinMaxScaler(Scaler):
    def __init__(self, scaled_min: int = 0, scaled_max: int = 1, epsilon: float = 1e-7):
        super().__init__()
        self.register_buffer('scaled_min', torch.tensor([scaled_min]), persistent=False)
        self.register_buffer('scaled_max', torch.tensor([scaled_max]), persistent=False)
        self.register_buffer('mins', Tensor(), persistent=False)
        self.register_buffer('maxs', Tensor(), persistent=False)
        self.register_buffer('epsilon', torch.tensor([epsilon]), persistent=False)

    def fit(self, values: Tensor):
        self._validate_input(values)
        if values.shape[0] <= 1:
            raise ValueError("Must have more than one sample to fit to, got " + str(values.shape[0]))
        self.mins = torch.min(values, dim=0).values
        self.maxs = torch.max(values, dim=0).values
        assert len(self.mins) == len(self.maxs) == values.shape[-1]

    def transform(self, values: Tensor):
        self._validate_input(values)
        standard_scale = (values - self.mins) / (self.maxs - self.mins + self.epsilon)
        return standard_scale * (self.scaled_max - self.scaled_min) + self.scaled_min

    def inverse_transform(self, values):
        self._validate_input(values)
        standard_scale = (values - self.scaled_min) / (self.scaled_max - self.scaled_min)
        return standard_scale * (self.maxs - self.mins + self.epsilon) + self.mins


def fit_label_scaler(dataset: Dataset, scaler_type: Type[Scaler]) -> Scaler:
    scaler = scaler_type()
    scaler.fit(dataset.data.y.reshape(-1, 1))
    return scaler
