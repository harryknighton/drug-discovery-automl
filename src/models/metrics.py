from typing import Any, List

import torch
from fast_pytorch_kmeans import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from torch import Tensor
from torchmetrics import PearsonCorrCoef, MeanSquaredError, Metric, MetricCollection, MeanAbsoluteError, R2Score

from src.config import AUTOML_LOGGER
from src.types import Metrics


# -----------------------------------------------
# Accuracy Metrics


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


# -----------------------------------------------
# Explainability Metrics


class ConceptCompleteness(Metric):
    """Compute the concept completeness adapted from GCExplainer (https://arxiv.org/abs/2107.11889)"""
    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("encodings", default=Tensor(), dist_reduce_fx="cat")
        self.add_state("targets", default=Tensor(), dist_reduce_fx="cat")

    def update(self, encoding: Tensor, target: Tensor) -> None:
        self.encodings = torch.cat((self.encodings, encoding), dim=0)
        self.targets = torch.cat((self.targets, target), dim=0)

    def compute(self) -> Tensor:
        cluster_labels = cluster_graphs(self.encodings).reshape(-1, 1).cpu().numpy()
        targets = self.targets.cpu().numpy()
        decision_tree = DecisionTreeRegressor()
        decision_tree.fit(cluster_labels, targets)
        predictions = decision_tree.predict(cluster_labels)
        return torch.tensor(mean_squared_error(predictions, targets))


def cluster_graphs(encodings: Tensor, max_clusters: int = 10) -> Tensor:
    cluster_labels = []
    silhouette_scores = []
    wss_scores = []
    # Calculate fitness scores for each choice of k
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k)
        labels = kmeans.fit_predict(encodings)
        wss_score = _within_cluster_sum_squared_error(encodings, kmeans.centroids, labels)
        silhouette = silhouette_score(encodings, labels)
        cluster_labels.append(labels)
        silhouette_scores.append(silhouette)
        wss_scores.append(wss_score)
    # Choose best fit as the one with maximum silhouette score
    best_index = torch.stack(silhouette_scores).argmax()
    _validate_kmeans(best_index, torch.stack(wss_scores))
    return cluster_labels[best_index]


def _within_cluster_sum_squared_error(encodings: Tensor, centroids: Tensor, labels: Tensor) -> Tensor:
    differences = encodings - centroids[labels]
    return torch.sum(differences ** 2)


def _validate_kmeans(best_index: int, wss_scores: Tensor, elbow_tolerance: float = 0.5) -> None:
    wss_gradients = torch.diff(wss_scores)
    if 0 < best_index < len(wss_scores) - 1:
        # Are we past the bend of the elbow?
        if wss_gradients[best_index - 1] * elbow_tolerance > wss_gradients[best_index]:
            AUTOML_LOGGER.warn('KMeans number of clusters may be too high')
    elif best_index == len(wss_scores) - 1:
        # Are we before the bend of the elbow?
        if wss_gradients[best_index - 2] * elbow_tolerance < wss_gradients[best_index - 1]:
            AUTOML_LOGGER.warn('KMeans number of clusters may be too low')


def silhouette_score(encodings: Tensor, cluster_labels: Tensor) -> Tensor:
    clusters_encodings = [encodings[cluster_labels == label] for label in cluster_labels.unique()]
    intra_cluster_distances = _intra_cluster_distances(clusters_encodings)
    inter_cluster_distances = _inter_cluster_distances(clusters_encodings)
    max_cluster_distances = torch.max(inter_cluster_distances, intra_cluster_distances)
    silhouette_scores = (inter_cluster_distances - intra_cluster_distances) / max_cluster_distances
    return torch.mean(torch.nan_to_num(silhouette_scores))


def _intra_cluster_distances(clusters_encodings: List[Tensor]) -> Tensor:
    return torch.cat([
        torch.cdist(cluster_encodings, cluster_encodings).sum(dim=1) / (cluster_encodings.shape[0] - 1)
        for cluster_encodings in clusters_encodings
    ])


def _inter_cluster_distances(clusters_encodings: List[Tensor]) -> Tensor:
    distances = [torch.full((encoding.size(0),), torch.inf, device=encoding.device) for encoding in clusters_encodings]
    for i, encodings_i in zip(range(len(clusters_encodings)), clusters_encodings):
        for j, encodings_j in zip(range(i), clusters_encodings):
            pairwise_distances = torch.cdist(encodings_i, encodings_j)
            distances[i] = torch.min(distances[i], pairwise_distances.mean(dim=1))
            distances[j] = torch.min(distances[j], pairwise_distances.mean(dim=0))
    return torch.cat(distances)


# -----------------------------------------------
# Utilities


DEFAULT_EXPLAINABILITY_METRICS = MetricCollection([ConceptCompleteness()])
DEFAULT_ACCURACY_METRICS = MetricCollection([
    MeanAbsoluteError(),
    RootMeanSquaredError(),
    MaxError(),
    PearsonCorrCoefSquared(),
    R2Score(),
])


def detach_metrics(metrics: Metrics) -> Metrics:
    return {k: v.detach() for k, v in metrics.items()}
