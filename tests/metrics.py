import pytest
import torch

from src.metrics import StandardScaler, MinMaxScaler, analyse_results_distribution


def test_standard_scaler_with_1d_input_raises_error():
    data = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    scaler = StandardScaler()
    with pytest.raises(ValueError):
        scaler.fit_transform(data)


def test_standard_scaler_with_one_sample_raises_error():
    data = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.float32)
    scaler = StandardScaler()
    with pytest.raises(ValueError):
        scaler.fit_transform(data)


def test_standard_scaler_inverse_transform():
    original = torch.tensor([[1, 2, 3, 4, 5], [1, 3, 5, 7, 9]], dtype=torch.float32)
    scaler = StandardScaler()
    transformed = scaler.fit_transform(original)
    original_transformed = scaler.inverse_transform(transformed)
    assert torch.allclose(original, original_transformed)


def test_minmax_scaler_inverse_transform():
    original = torch.tensor([[1, 2, 3, 4, 5], [1, 3, 5, 7, 9]], dtype=torch.float32)
    scaler = MinMaxScaler()
    transformed = scaler.fit_transform(original)
    original_transformed = scaler.inverse_transform(transformed)
    assert torch.allclose(original, original_transformed)


def test_minmax_scaler_transforms_to_correct_bounds():
    original = torch.tensor([[1, 2, 3, 4, 5], [1, 3, 5, 7, 9]], dtype=torch.float32)
    scaler = MinMaxScaler(scaled_min=2, scaled_max=5)
    transformed = scaler.fit_transform(original)
    assert transformed.min() == 2 and transformed.max() == 5


def test_analyse_distribution_of_empty_results_raises_error():
    data = []
    with pytest.raises(AssertionError):
        analyse_results_distribution(data)


def test_analyse_results_distribution_returns_correctly():
    data = [
        {'mae': 1, 'r2': 4},
        {'mae': 4, 'r2': 6},
        {'mae': 2, 'r2': 8},
        {'mae': 3, 'r2': 2},
        {'mae': 5, 'r2': 0},
    ]
    results = analyse_results_distribution(data)
    assert 'mae' in results and 'r2' in results
    assert (
        results['mae']['min'] == 1 and
        results['mae']['p25'] == 2 and
        results['mae']['median'] == 3 and
        results['mae']['p75'] == 4 and
        results['mae']['max'] == 5
    )
    assert (
        results['r2']['min'] == 0 and
        results['r2']['p25'] == 2 and
        results['r2']['median'] == 4 and
        results['r2']['p75'] == 6 and
        results['r2']['max'] == 8
    )
