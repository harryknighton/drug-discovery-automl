from abc import ABC
from typing import List

import torch
import torch_geometric.nn
from torch import Tensor
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader

from src.models import GNNModule


class Proxy(ABC):
    higher_is_better = False

    def __init__(self, num_samples: int = 64) -> None:
        self.num_samples = num_samples

    def calculate(self, model: GNNModule, dataset: Dataset) -> float:
        pass

    def __call__(self, model: GNNModule, dataset: Dataset) -> float:
        return self.calculate(model, dataset)


class MajorityVote(Proxy):
    def __init__(self, proxies: List[Proxy]) -> None:
        super(Proxy, self).__init__()
        self.proxies = proxies

    def calculate(self, model: GNNModule, dataset: Dataset) -> float:
        results = [proxy(model, dataset) for proxy in self.proxies]
        return sum(results) / len(results)


# ----------------------------------------------
# Data Independent Proxies


class NumParams(Proxy):
    def calculate(self, model: GNNModule, dataset: Dataset) -> float:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SynapticFlow(Proxy):
    def calculate(self, model: GNNModule, dataset: Dataset) -> float:
        pass

# ----------------------------------------------
# Data Dependent Proxies


class JacobianCovariance(Proxy):
    higher_is_better = True

    def __init__(self, num_samples: int = 64) -> None:
        super(JacobianCovariance, self).__init__()
        self.num_samples = num_samples

    def calculate(self, model: GNNModule, dataset: Dataset) -> float:
        batch = next(iter(DataLoader(dataset, batch_size=self.num_samples, shuffle=False)))
        jacobian = self._compute_jacobian(model, batch)
        _, log_determinant = torch.linalg.slogdet(jacobian)
        return log_determinant

    @staticmethod
    def _compute_jacobian(model: GNNModule, batch: Data) -> Tensor:
        batch.x.requires_grad_(True)
        preds = model(batch.xs, batch.edge_index, batch.batch)
        preds.backward(torch.ones_like(preds))
        jacob = batch.x.grad.detach()
        batch.x.requires_grad_(False)
        return jacob


class ZiCo(Proxy):
    def calculate(self, model: GNNModule, dataset: Dataset) -> float:
        pass


class GradientNorm(Proxy):
    def calculate(self, model: GNNModule, dataset: Dataset) -> float:
        batch = next(iter(DataLoader(dataset, batch_size=self.num_samples, shuffle=False)))
        model.train()
        preds = model(batch.xs, batch.edge_index, batch.batch)
        preds.backward()
        gradient_norm = torch.tensor(0)
        for layer in model.modules():
            if isinstance(layer, (torch.nn.Linear, torch_geometric.nn.Linear)) and layer.weight.grad is not None:
                gradient_norm += layer.weight.grad.norm()
        return gradient_norm.detach()


class Snip(Proxy):
    def calculate(self, model: GNNModule, dataset: Dataset) -> float:
        pass
