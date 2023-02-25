from abc import ABC
from typing import Any, List

import numpy as np
import torch
from torch import Tensor
from torchmetrics import PearsonCorrCoef, MeanSquaredError, Metric, MetricCollection, MeanAbsoluteError, R2Score


class PearsonCorrCoefSquared(PearsonCorrCoef):
    """Provides an alternative implementation of R^2"""

    def compute(self) -> Tensor:
        r = super(PearsonCorrCoefSquared, self).compute()
        return torch.pow(r, 2)


class RootMeanSquaredError(MeanSquaredError):
    def __init__(self):
        super(RootMeanSquaredError, self).__init__(squared=False)


class MaxError(Metric):
    """Computes the maximum error of any sample"""
    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False
    max_error: Tensor = -1.0

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("max_error", default=torch.tensor(-1.0), dist_reduce_fx="max")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        batch_max_error = (preds - target).abs().max()
        self.max_error = max(self.max_error, batch_max_error)

    def compute(self) -> Tensor:
        return self.max_error


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


DEFAULT_METRICS = MetricCollection([
    MeanAbsoluteError(),
    RootMeanSquaredError(),
    MaxError(),
    PearsonCorrCoefSquared(),
    R2Score(),
])


def analyse_results_distribution(results: List[dict[str, float]]) -> dict[str, dict]:
    """Calculate the distribution of the results of all trials"""
    assert results is not None and len(results) > 0
    stacked_metrics = {metric: np.array([float(result[metric]) for result in results]) for metric in results[0]}
    metrics = {}
    for metric, values in stacked_metrics.items():
        percentiles = np.percentile(values, [0, 25, 50, 75, 100])
        variance = np.var(values, ddof=1) if len(values) > 1 else 0.0
        metrics[metric] = {
            'mean': np.mean(values),
            'variance': variance,
            'min': percentiles[0],
            'p25': percentiles[1],
            'median': percentiles[2],
            'p75': percentiles[3],
            'max': percentiles[4]
        }
    return metrics
