import pytest

from src.models import ActivationFunction, ModelArchitecture, GNNLayerType


def test_invalid_model_architecture_raises_error():
    with pytest.raises(AssertionError):
        ModelArchitecture(
            layer_types=[GNNLayerType.GCN],
            features=[113, 113, 113, 1],
            activation_funcs=[ActivationFunction.ReLU],
            batch_normalise=[True, False],
        )
