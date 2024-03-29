"""Implement zero-cost proxies for use in NAS

Copyright (c) 2023, Harry Knighton
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

import copy
from abc import ABC
from typing import List

import torch
import torch_geometric.nn
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from torch import Tensor, autograd
from torch.nn import Module
from torch.nn.functional import mse_loss, pdist
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool

from src.config import DEFAULT_PROXY_BATCH_SIZE
from src.data.utils import NamedLabelledDataset
from src.models.metrics import cluster_graphs
from src.models.gnns import GNNModule, GNN
from src.types import Metrics


class Proxy(ABC):
    """Define proxy to approximate target metric from model"""
    def __init__(self) -> None:
        self.num_samples = DEFAULT_PROXY_BATCH_SIZE
        self.higher_is_better = None

    def _compute(self, model: GNNModule, dataset: NamedLabelledDataset) -> Tensor:
        pass

    def __call__(self, model: GNNModule, dataset: NamedLabelledDataset) -> Tensor:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model_copy = copy.deepcopy(model).to(device)
        model_copy.zero_grad()
        model_copy.train()
        return self._compute(model_copy, dataset).detach().cpu()

    def fit(self, xs: Metrics, y: Tensor, maximise_label: bool) -> None:
        x = xs[self.__class__.__name__]
        correlation = pearsonr(x.cpu(), y.cpu()).statistic
        self.higher_is_better = correlation < 0 and not maximise_label or correlation > 0 and maximise_label


class ProxyCollection(Proxy):
    """Collate proxies and calculate all of them."""
    def __init__(self, proxies: List[Proxy]) -> None:
        super(ProxyCollection, self).__init__()
        self.proxies = {proxy.__class__.__name__: proxy for proxy in proxies}

    def _compute(self, model: GNNModule, dataset: NamedLabelledDataset) -> Metrics:
        return {name: proxy(model, dataset) for name, proxy in self.proxies.items()}

    def __call__(self, model: GNNModule, dataset: NamedLabelledDataset) -> Metrics:
        return self._compute(model, dataset)

    def fit(self, xs: Metrics, y: Tensor, maximise_label: bool) -> None:
        for proxy in self.proxies.values():
            proxy.fit(xs, y, maximise_label)


class Ensemble(Proxy):
    """Combine proxies using a linear regression model."""
    def __init__(self, proxy_collection: ProxyCollection) -> None:
        super(Proxy, self).__init__()
        self.proxy_collection = proxy_collection
        self.model = LinearRegression()

    def _compute(self, model: GNNModule, dataset: NamedLabelledDataset) -> Tensor:
        proxies = self.proxy_collection(model, dataset)
        x = torch.stack(list(proxies.values())).reshape(1, -1)
        result = self.model.predict(x)
        return torch.tensor(result)

    def fit(self, xs: Metrics, y: Tensor, maximise_label: bool) -> None:
        self.proxy_collection.fit(xs, y, maximise_label)
        self.higher_is_better = maximise_label
        x = torch.stack([xs[proxy] for proxy in self.proxy_collection.proxies], dim=1)
        self.model.fit(x.cpu(), y.cpu())


# ----------------------------------------------
# Data Independent Proxies


class NumParams(Proxy):
    def _compute(self, model: GNNModule, dataset: NamedLabelledDataset) -> Tensor:
        return torch.tensor(
            sum(p.numel() for p in model.parameters() if p.requires_grad),
            dtype=torch.float, device=next(model.parameters()).device
        )


class SynFlow(Proxy):
    """Implement SynFlow

    References:
        Abdelfattah et al. Zero-Cost Proxies for Lightweight NAS
        https://arxiv.org/abs/2101.08134
    """
    def _compute(self, model: GNNModule, dataset: NamedLabelledDataset) -> Tensor:
        weights = _get_model_weights(model)
        preds = model(
            x=torch.ones(1, dataset.dataset.num_features, dtype=torch.float, device=weights[0].device),
            edge_index=torch.tensor([[0], [0]], dtype=torch.long, device=weights[0].device),
            batch=torch.tensor([0], dtype=torch.long, device=weights[0].device)
        )
        loss = preds.sum()
        first_derivatives = autograd.grad(loss, weights, allow_unused=True)
        synflow_per_weight = [(weight * coefficients).sum() for weight, coefficients in zip(weights, first_derivatives)]
        return sum(synflow_per_weight)

# ----------------------------------------------
# Data Dependent Proxies


class JacobianCovariance(Proxy):
    """Implement JacobCov from 'Neural Architecture Search without Training'

    References:
        Joseph Mellor et al. Neural Architecture Search without Training
        https://arxiv.org/abs/2006.04647
    """
    def _compute(self, model: GNNModule, dataset: NamedLabelledDataset) -> Tensor:
        jacobian = self._compute_batch_jacobian(model, dataset)
        correlations = torch.corrcoef(jacobian)
        sign, log_determinant = torch.slogdet(correlations)
        if sign == 0:
            return torch.tensor(0.)
        else:
            return log_determinant

    def _compute_batch_jacobian(self, model: GNNModule, dataset: NamedLabelledDataset) -> Tensor:
        batch = _get_data_samples(model, dataset.dataset, self.num_samples)
        batch.x.requires_grad_(True)
        preds = model(batch.x, batch.edge_index, batch.batch)
        jacob = autograd.grad(preds, batch.x, torch.ones_like(preds))[0]
        batch.x.requires_grad_(False)
        graph_jacob = global_add_pool(jacob, batch.batch)
        return graph_jacob


class GradientNorm(Proxy):
    """Implement GradNorm

    References:
        Abdelfattah et al. Zero-Cost Proxies for Lightweight NAS
        https://arxiv.org/abs/2101.08134
    """
    def _compute(self, model: GNNModule, dataset: NamedLabelledDataset) -> Tensor:
        batch = _get_data_samples(model, dataset.dataset, self.num_samples)
        preds = model(batch.x, batch.edge_index, batch.batch)
        scaled_preds = dataset.label_scaler.to(preds.device).inverse_transform(preds)
        loss = mse_loss(scaled_preds.flatten(), batch.y)
        weights = _get_model_weights(model)
        gradients = autograd.grad(loss, weights)
        return sum(gradient.sum() for gradient in gradients)


class Snip(Proxy):
    """Implement Snip zero-cost proxy

    References:
        Abdelfattah et al. Zero-Cost Proxies for Lightweight NAS
        https://arxiv.org/abs/2101.08134
    """
    def _compute(self, model: GNNModule, dataset: NamedLabelledDataset) -> Tensor:
        batch = _get_data_samples(model, dataset.dataset, self.num_samples)
        weights = _get_model_weights(model)
        preds = model(batch.x, batch.edge_index, batch.batch)
        scaled_preds = dataset.label_scaler.to(preds.device).inverse_transform(preds)
        loss = mse_loss(scaled_preds.flatten(), batch.y)
        first_derivatives = autograd.grad(loss, weights, allow_unused=True)
        snip_per_weight = [(weight * coefficients).abs().sum() for weight, coefficients in zip(weights, first_derivatives)]
        return sum(snip_per_weight)


class ZiCo(Proxy):
    """Implement ZiCo zero-cost proxy

    References:
        Li et al. ZiCo: Zero-shot NAS via Inverse Coefficient of Variation on Gradients
        https://arxiv.org/abs/2301.11300
    """
    def _compute(self, model: GNNModule, dataset: NamedLabelledDataset) -> Tensor:
        batch = _get_data_samples(model, dataset.dataset, self.num_samples)
        weights = _get_model_weights(model)
        preds = model(batch.x, batch.edge_index, batch.batch)
        scaled_preds = dataset.label_scaler.to(preds.device).inverse_transform(preds)
        loss = mse_loss(scaled_preds.flatten(), batch.y)
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


class Grasp(Proxy):
    """Implement Grasp zero-cost proxy

    References:
        Abdelfattah et al. Zero-Cost Proxies for Lightweight NAS
        https://arxiv.org/abs/2101.08134
    """
    def _compute(self, model: GNNModule, dataset: NamedLabelledDataset) -> Tensor:
        batch = _get_data_samples(model, dataset.dataset, self.num_samples)
        weights = _get_model_weights(model)
        preds = model(batch.x, batch.edge_index, batch.batch)
        scaled_preds = dataset.label_scaler.to(preds.device).inverse_transform(preds)
        loss = mse_loss(scaled_preds.flatten(), batch.y)
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
    """Implement Fisher zero-cost proxy

    References:
        Abdelfattah et al. Zero-Cost Proxies for Lightweight NAS
        https://arxiv.org/abs/2101.08134
    """
    def _compute(self, model: GNNModule, dataset: NamedLabelledDataset) -> Tensor:
        # Attach forward hook to capture activations
        def activation_hook(module: Module, _: Tensor, output: Tensor) -> None:
            module.activation = output
        linear_layers = _get_linear_layers(model)
        hook_handles = [layer.register_forward_hook(activation_hook) for layer in linear_layers]

        # Run data through model to calculate activations
        batch = _get_data_samples(model, dataset.dataset, self.num_samples)
        preds = model(batch.x, batch.edge_index, batch.batch)
        scaled_preds = dataset.label_scaler.to(preds.device).inverse_transform(preds)
        loss = mse_loss(scaled_preds.flatten(), batch.y)

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

# ----------------------------------------------
# Novel explainability-oriented proxies


class LatentSparsityGrad(Proxy):
    def _compute(self, model: GNN, dataset: NamedLabelledDataset) -> Tensor:
        # Calculate impurity of the latent space
        batch = _get_data_samples(model, dataset.dataset, self.num_samples)
        encodings = model.encode(batch.x, batch.edge_index, batch.batch)
        latent_distances = pdist(encodings)
        sparsity = torch.mean(latent_distances)

        # Calculate proxy as gradient of impurity
        weights = _get_model_weights(model)
        gradients = autograd.grad(sparsity, weights, allow_unused=True)
        saliency_per_layer = [
            (weight * gradient).abs().sum()
            for weight, gradient in zip(weights, gradients)
            if gradient is not None  # Ignore readout layers
        ]
        return sum(saliency_per_layer)


class ConceptPurityGrad(Proxy):
    def _compute(self, model: GNN, dataset: NamedLabelledDataset) -> Tensor:
        # Calculate impurity of the latent space
        batch = _get_data_samples(model, dataset.dataset, self.num_samples)
        encodings = model.encode(batch.x, batch.edge_index, batch.batch)
        cluster_labels = cluster_graphs(encodings)
        latent_distances = [pdist(encodings[cluster_labels == c]) for c in cluster_labels.unique()]
        label_distances = [pdist(batch.y[cluster_labels == c].reshape(-1, 1)) for c in cluster_labels.unique()]
        concept_impurities = [
            torch.mean(latent_ds / (1 + label_ds)) if len(label_ds) > 0 else 0.
            for latent_ds, label_ds in zip(latent_distances, label_distances)
        ]
        impurity = sum(concept_impurities) / len(concept_impurities)

        # Calculate proxy as gradient of impurity
        weights = _get_model_weights(model)
        gradients = autograd.grad(impurity, weights, allow_unused=True)
        saliency_per_layer = [
            (weight * gradient).abs().sum()
            for weight, gradient in zip(weights, gradients)
            if gradient is not None  # Ignore readout layers
        ]
        return sum(saliency_per_layer)


DEFAULT_PROXIES = ProxyCollection([
    NumParams(),
    SynFlow(),
    GradientNorm(),
    JacobianCovariance(),
    Snip(),
    ZiCo(),
    Grasp(),
    Fisher(),
    LatentSparsityGrad(),
    ConceptPurityGrad(),
])


def _get_data_samples(model: GNNModule, dataset: Dataset, num_samples: int) -> Data:
    device = next(model.parameters()).device
    data = next(iter(DataLoader(dataset, batch_size=num_samples, shuffle=True)))
    return data.to(device)


def _get_model_weights(model: GNNModule) -> List[Tensor]:
    return [layer.weight for layer in _get_linear_layers(model)]


def _get_linear_layers(model: GNNModule) -> List[torch.nn.Linear | torch_geometric.nn.Linear]:
    return [
        module for module in model.modules()
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch_geometric.nn.Linear)
    ]
