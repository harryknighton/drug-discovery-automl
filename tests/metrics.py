import pytest
import torch

from src.metrics import StandardScaler


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
