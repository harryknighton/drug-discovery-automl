import pytest
import torch

from src.models.metrics import silhouette_score


def test_silhouette_score_with_only_one_cluster_is_0():
    encodings = torch.rand((3, 10))
    cluster_labels = torch.zeros(3)
    expected = 0.
    result = silhouette_score(encodings, cluster_labels)
    assert expected == result


def test_silhouette_score_with_two_one_element_clusters_is_0():
    encodings = torch.rand((2, 10))
    cluster_labels = torch.tensor([0, 1])
    expected = 0.
    result = silhouette_score(encodings, cluster_labels)
    assert expected == result


def test_silhouette_score_with_one_cluster_of_multiple_of_the_same_point_is_0():
    encodings = torch.rand((1, 10)).repeat(3, 1)
    cluster_labels = torch.zeros(3)
    expected = 0.
    result = silhouette_score(encodings, cluster_labels)
    assert expected == result


def test_silhouette_score_with_differently_sized_inputs_raises_error():
    encodings = torch.rand((2, 10))
    cluster_labels = torch.zeros(4)
    with pytest.raises(ValueError):
        silhouette_score(encodings, cluster_labels)
