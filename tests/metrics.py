import pytest
from torch import tensor

from src.evaluation.metrics import analyse_results_distribution


def test_analyse_distribution_of_empty_results_raises_error():
    data = {}
    with pytest.raises(AssertionError):
        analyse_results_distribution(data)


def test_analyse_results_distribution_returns_correctly():
    data = {
        0: {'mae': tensor(1), 'r2': tensor(4)},
        1: {'mae': tensor(4), 'r2': tensor(6)},
        2: {'mae': tensor(2), 'r2': tensor(8)},
        3: {'mae': tensor(3), 'r2': tensor(2)},
        4: {'mae': tensor(5), 'r2': tensor(0)},
    }
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
