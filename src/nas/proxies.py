from abc import ABC
from typing import List

import torch
import torch_geometric.nn
from torch import Tensor
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool

from src.config import DEFAULT_PROXY_BATCH_SIZE
from src.models import GNNModule


class Proxy(ABC):
    higher_is_better = True

    def __init__(self) -> None:
        self.num_samples = DEFAULT_PROXY_BATCH_SIZE

    def compute(self, model: GNNModule, dataset: Dataset) -> Tensor:
        pass

    def __call__(self, model: GNNModule, dataset: Dataset) -> Tensor:
        return self.compute(model, dataset)


class ProxyCollection(Proxy):
    def __init__(self, proxies: List[Proxy]) -> None:
        super(ProxyCollection, self).__init__()
        self.proxies = proxies

    def compute(self, model: GNNModule, dataset: Dataset) -> dict[str, float]:
        return {proxy.__class__.__name__: proxy(model, dataset) for proxy in self.proxies}


class MajorityVote(Proxy):
    def __init__(self, proxies: List[Proxy]) -> None:
        super(Proxy, self).__init__()
        self.proxies = proxies

    def compute(self, model: GNNModule, dataset: Dataset) -> Tensor:
        results = [proxy(model, dataset) for proxy in self.proxies]
        return sum(results) / len(results)


# ----------------------------------------------
# Data Independent Proxies


class NumParams(Proxy):
    def compute(self, model: GNNModule, dataset: Dataset) -> Tensor:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SynapticFlow(Proxy):
    def compute(self, model: GNNModule, dataset: Dataset) -> Tensor:
        pass

# ----------------------------------------------
# Data Dependent Proxies


class JacobianCovariance(Proxy):
    def compute(self, model: GNNModule, dataset: Dataset) -> Tensor:
        batch = _get_data_samples(dataset, self.num_samples)
        jacobian = self._compute_jacobian(model, batch)
        correlations = torch.corrcoef(jacobian)
        _, logdet = torch.slogdet(correlations)
        return logdet

    @staticmethod
    def _compute_jacobian(model: GNNModule, data: Data) -> Tensor:
        data.x.requires_grad_(True)
        preds = model(data.x, data.edge_index, data.batch)
        preds.backward(torch.ones_like(preds))
        jacob = data.x.grad.detach()
        data.x.requires_grad_(False)
        graph_jacob = global_add_pool(jacob, data.batch)
        return graph_jacob


class GradientNorm(Proxy):
    def compute(self, model: GNNModule, dataset: Dataset) -> Tensor:
        batch = _get_data_samples(dataset, self.num_samples)
        model.train()
        preds = model(batch.x, batch.edge_index, batch.batch)
        preds.backward(torch.ones_like(preds))
        gradient_norm = torch.tensor(0, dtype=torch.float)
        for layer in model.modules():
            if isinstance(layer, (torch.nn.Linear, torch_geometric.nn.Linear)) and layer.weight.grad is not None:
                gradient_norm += layer.weight.grad.norm()
        return gradient_norm.detach()


class Snip(Proxy):
    def compute(self, model: GNNModule, dataset: Dataset) -> Tensor:
        pass


class Grasp(Proxy):
    def compute(self, model: GNNModule, dataset: Dataset) -> Tensor:
        pass


class Fisher(Proxy):
    def compute(self, model: GNNModule, dataset: Dataset) -> Tensor:
        pass


class ZiCo(Proxy):
    def compute(self, model: GNNModule, dataset: Dataset) -> Tensor:
        pass


class NASI(Proxy):
    def compute(self, model: GNNModule, dataset: Dataset) -> Tensor:
        pass


DEFAULT_PROXIES = ProxyCollection([
    NumParams(),
    # SynapticFlow(),
    GradientNorm(),
    JacobianCovariance(),
    # Snip(),
    # Grasp(),
    # Fisher()
])


def _get_data_samples(dataset: Dataset, num_samples: int) -> Data:
    return next(iter(DataLoader(dataset, batch_size=num_samples, shuffle=False)))
