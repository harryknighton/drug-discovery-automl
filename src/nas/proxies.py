from abc import ABC
from typing import List

import torch
import torch_geometric.nn
from torch import Tensor, autograd
from torch.nn.functional import mse_loss
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


class SynFlow(Proxy):
    def compute(self, model: GNNModule, dataset: Dataset) -> Tensor:
        weights = _get_model_weights(model)
        preds = model(
            x=torch.ones(1, dataset.num_features, dtype=torch.float),
            edge_index=torch.tensor([[0], [0]], dtype=torch.long),
            batch=torch.tensor([0], dtype=torch.long)
        )
        loss = preds.sum()
        first_derivatives = autograd.grad(loss, weights, allow_unused=True)
        synflow_per_weight = [(weight * coefficients).sum() for weight, coefficients in zip(weights, first_derivatives)]
        return sum(synflow_per_weight)

# ----------------------------------------------
# Data Dependent Proxies


class JacobianCovariance(Proxy):
    def compute(self, model: GNNModule, dataset: Dataset) -> Tensor:
        batch = _get_data_samples(dataset, self.num_samples)
        jacobian = self._compute_jacobian(model, batch)
        correlations = torch.corrcoef(jacobian)
        _, log_determinant = torch.slogdet(correlations)
        return log_determinant

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
        batch = _get_data_samples(dataset, self.num_samples)
        model.train()
        weights = _get_model_weights(model)
        preds = model(batch.x, batch.edge_index, batch.batch)
        loss = mse_loss(preds.flatten(), batch.y)
        first_derivatives = autograd.grad(loss, weights, allow_unused=True)
        snip_per_weight = [(weight * coefficients).abs().sum() for weight, coefficients in zip(weights, first_derivatives)]
        return sum(snip_per_weight)


class ZiCo(Proxy):
    def compute(self, model: GNNModule, dataset: Dataset) -> Tensor:
        raise NotImplementedError()


class NASI(Proxy):
    def compute(self, model: GNNModule, dataset: Dataset) -> Tensor:
        raise NotImplementedError()


class Grasp(Proxy):
    def compute(self, model: GNNModule, dataset: Dataset) -> Tensor:
        batch = _get_data_samples(dataset, self.num_samples)
        model.train()
        weights = _get_model_weights(model)
        preds = model(batch.x, batch.edge_index, batch.batch)
        loss = mse_loss(preds.flatten(), batch.y)
        # [dL/dw_i for i in len(weights)]
        first_derivatives = autograd.grad(loss, weights, create_graph=True, allow_unused=True)
        # (dL/dw_0)^2 + (dL/dw_1)^2 + ... + (dL/dw_n)^2
        sum_derivatives_squared = sum([(derivative ** 2).sum() for derivative in first_derivatives])
        # [sum(dL^2/d^2w_iw_j dL/dw_i for i in len(weights)) for j in len(weights)]
        hessian_jacob_prod = autograd.grad(sum_derivatives_squared, weights, allow_unused=True)
        # [Hessian * dL/dW * W for W in weights]
        grasp_per_weight = [(weight * coefficients).sum() for weight, coefficients in zip(weights, hessian_jacob_prod)]
        return sum(grasp_per_weight)


class Fisher(Proxy):
    def compute(self, model: GNNModule, dataset: Dataset) -> Tensor:
        raise NotImplementedError()


DEFAULT_PROXIES = ProxyCollection([
    NumParams(),
    SynFlow(),
    GradientNorm(),
    JacobianCovariance(),
    Snip(),
    # ZiCo(),
    Grasp(),
    # Fisher()
])


def _get_data_samples(dataset: Dataset, num_samples: int) -> Data:
    return next(iter(DataLoader(dataset, batch_size=num_samples, shuffle=False)))


def _get_model_weights(model: GNNModule) -> List[Tensor]:
    weights = []
    for module in model.modules():
        if hasattr(module, 'weight') and module.weight.requires_grad:
            weights.append(module.weight)
    return weights
