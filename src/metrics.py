from typing import Any, Optional

import torch
from torch import Tensor
from torchmetrics import PearsonCorrCoef, MeanSquaredError, Metric, MetricCollection, MeanAbsoluteError, R2Score

from src.data import HTSDataset


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


class StandardScaler:
    def __init__(self, mean: Optional[int] = None, std: Optional[int] = None, epsilon: float = 1e-7):
        """Standard Scaler for use with Pytorch Tensors on GPU

        Adapted from https://gist.github.com/farahmand-m/8a416f33a27d73a149f92ce4708beb40
        """
        self.means = None
        self.stds = None
        self.epsilon = epsilon

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

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    def inverse_transform(self, values):
        self._validate_input(values)
        return values * (self.stds + self.epsilon) + self.means

    @staticmethod
    def _validate_input(values):
        if values.dim() != 2:
            raise ValueError("Values must be shape (n_samples, n_features) but got " + str(values.shape))



DEFAULT_METRICS = MetricCollection([
    MeanAbsoluteError(),
    RootMeanSquaredError(),
    MaxError(),
    PearsonCorrCoefSquared(),
    R2Score(),
])
