from typing import Any

import torch
from fast_pytorch_kmeans import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from torch import Tensor
from torchmetrics import Metric, MetricCollection

from src.config import AUTOML_LOGGER
from src.models.evaluation.pytorchclustermetrics import silhouette_score


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


DEFAULT_EXPLAINABILITY_METRICS = MetricCollection([ConceptCompleteness()])


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
