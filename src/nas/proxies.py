import copy
from abc import ABC
from typing import List

import torch
import torch_geometric.nn
from torch import Tensor, autograd
from torch.nn import Module
from torch.nn.functional import mse_loss
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool

from src.config import DEFAULT_PROXY_BATCH_SIZE
from src.models import GNNModule
from src.types import Metrics


class Proxy(ABC):
    higher_is_better = True

    def __init__(self) -> None:
        self.num_samples = DEFAULT_PROXY_BATCH_SIZE

    def _compute(self, model: GNNModule, dataset: Dataset) -> Tensor:
        pass

    def __call__(self, model: GNNModule, dataset: Dataset) -> Tensor:
        model_copy = copy.deepcopy(model)
        model_copy.zero_grad()
        model_copy.train()
        return self._compute(model_copy, dataset).detach()


class ProxyCollection(Proxy):
    def __init__(self, proxies: List[Proxy]) -> None:
        super(ProxyCollection, self).__init__()
        self.proxies = proxies

    def _compute(self, model: GNNModule, dataset: Dataset) -> Metrics:
        return {proxy.__class__.__name__: proxy(model, dataset) for proxy in self.proxies}

    def __call__(self, model: GNNModule, dataset: Dataset) -> Metrics:
        return self._compute(model, dataset)


class MajorityVote(Proxy):
    def __init__(self, proxies: List[Proxy]) -> None:
        super(Proxy, self).__init__()
        self.proxies = proxies

    def _compute(self, model: GNNModule, dataset: Dataset) -> Tensor:
        results = [proxy(model, dataset) for proxy in self.proxies]
        return sum(results) / len(results)


# ----------------------------------------------
# Data Independent Proxies


class NumParams(Proxy):
    def _compute(self, model: GNNModule, dataset: Dataset) -> Tensor:
        return torch.tensor(sum(p.numel() for p in model.parameters() if p.requires_grad))


class SynFlow(Proxy):
    def _compute(self, model: GNNModule, dataset: Dataset) -> Tensor:
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
    def _compute(self, model: GNNModule, dataset: Dataset) -> Tensor:
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
    def _compute(self, model: GNNModule, dataset: Dataset) -> Tensor:
        batch = _get_data_samples(dataset, self.num_samples)
        preds = model(batch.x, batch.edge_index, batch.batch)
        preds.backward(torch.ones_like(preds))
        gradient_norm = torch.tensor(0, dtype=torch.float)
        for layer in model.modules():
            if isinstance(layer, (torch.nn.Linear, torch_geometric.nn.Linear)) and layer.weight.grad is not None:
                gradient_norm += layer.weight.grad.norm()
        return gradient_norm.detach()


class Snip(Proxy):
    def _compute(self, model: GNNModule, dataset: Dataset) -> Tensor:
        batch = _get_data_samples(dataset, self.num_samples)
        weights = _get_model_weights(model)
        preds = model(batch.x, batch.edge_index, batch.batch)
        loss = mse_loss(preds.flatten(), batch.y)
        first_derivatives = autograd.grad(loss, weights, allow_unused=True)
        snip_per_weight = [(weight * coefficients).abs().sum() for weight, coefficients in zip(weights, first_derivatives)]
        return sum(snip_per_weight)


class ZiCo(Proxy):
    def _compute(self, model: GNNModule, dataset: Dataset) -> Tensor:
        batch = _get_data_samples(dataset, self.num_samples)
        weights = _get_model_weights(model)
        preds = model(batch.x, batch.edge_index, batch.batch)
        loss = mse_loss(preds.flatten(), batch.y)
        gradients = autograd.grad(loss, weights, allow_unused=True)
        gradient_means = [gradient.abs().mean(dim=0) for gradient in gradients]
        gradient_stds = [gradient.std(dim=0).nan_to_num() for gradient in gradients]
        non_zero_idxs = [gradient_std != 0 for gradient_std in gradient_stds]
        layer_inverse_coefficients_of_variation = [
            (means[mask] / stds[mask]).sum().log()
            for means, stds, mask in zip(gradient_means, gradient_stds, non_zero_idxs)
            if mask.any()
        ]
        return sum(layer_inverse_coefficients_of_variation)


class NASI(Proxy):
    def _compute(self, model: GNNModule, dataset: Dataset) -> Tensor:
        raise NotImplementedError()


class Grasp(Proxy):
    def _compute(self, model: GNNModule, dataset: Dataset) -> Tensor:
        batch = _get_data_samples(dataset, self.num_samples)
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
    def _compute(self, model: GNNModule, dataset: Dataset) -> Tensor:
        # Attach forward hook to capture activations
        def activation_hook(module: Module, _: Tensor, output: Tensor) -> None:
            module.activation = output
        linear_layers = _get_linear_layers(model)
        hook_handles = [layer.register_forward_hook(activation_hook) for layer in linear_layers]

        # Run data through model to calculate activations
        batch = _get_data_samples(dataset, self.num_samples)
        preds = model(batch.x, batch.edge_index, batch.batch)
        loss = mse_loss(preds.flatten(), batch.y)

        # Fetch activations and gradients of the loss w.r.t to them
        activations = [layer.activation for layer in linear_layers]
        gradients = autograd.grad(loss, activations)
        saliency_per_activation = []
        for activation, gradient in zip(activations, gradients):
            if activation.size(0) > self.num_samples:  # Activation is before pooling
                activation = global_add_pool(activation, batch.batch)
                gradient = global_add_pool(gradient, batch.batch)
            saliency = torch.pow(gradient * activation, 2).mean(0).sum()
            saliency_per_activation.append(saliency)

        # Restore state of model
        for handle in hook_handles:
            handle.remove()
        for layer in linear_layers:
            del layer.activation
        return sum(saliency_per_activation)


DEFAULT_PROXIES = ProxyCollection([
    NumParams(),
    SynFlow(),
    GradientNorm(),
    JacobianCovariance(),
    Snip(),
    ZiCo(),
    Grasp(),
    Fisher()
])


def _get_data_samples(dataset: Dataset, num_samples: int) -> Data:
    return next(iter(DataLoader(dataset, batch_size=num_samples, shuffle=False)))


def _get_model_weights(model: GNNModule) -> List[Tensor]:
    return [layer.weight for layer in _get_linear_layers(model)]


def _get_linear_layers(model: GNNModule) -> List[torch.nn.Linear | torch_geometric.nn.Linear]:
    return [
        module for module in model.modules()
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch_geometric.nn.Linear)
    ]



