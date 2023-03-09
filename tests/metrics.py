import pytest

from src.metrics import analyse_results_distribution


def test_analyse_distribution_of_empty_results_raises_error():
    data = {}
    with pytest.raises(AssertionError):
        analyse_results_distribution(data)


def test_analyse_results_distribution_returns_correctly():
    data = {
        0: {'mae': 1, 'r2': 4},
        1: {'mae': 4, 'r2': 6},
        2: {'mae': 2, 'r2': 8},
        3: {'mae': 3, 'r2': 2},
        4: {'mae': 5, 'r2': 0},
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
