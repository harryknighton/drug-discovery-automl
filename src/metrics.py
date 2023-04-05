from typing import Any

import numpy as np
import torch
from torch import Tensor
from torchmetrics import PearsonCorrCoef, MeanSquaredError, Metric, MetricCollection, MeanAbsoluteError, R2Score

from src.types import Metrics


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


DEFAULT_METRICS = MetricCollection([
    MeanAbsoluteError(),
    RootMeanSquaredError(),
    MaxError(),
    PearsonCorrCoefSquared(),
    R2Score(),
])


def analyse_results_distribution(results: dict[str | int, Metrics]) -> dict[str, Metrics]:
    """Calculate the distribution of results over all random seeds and data splits for a run"""
    assert len(results) > 0
    run_metrics = list(results.values())
    stacked_metrics = {
        metric: np.array([float(metrics[metric]) for metrics in run_metrics])
        for metric in run_metrics[0]
    }
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
