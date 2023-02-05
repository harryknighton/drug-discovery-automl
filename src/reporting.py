import json
import pickle
from datetime import datetime
from typing import Any

import pandas as pd
import torch
from torch import Tensor
from torchmetrics import PearsonCorrCoef, MeanSquaredError, Metric

from src.data import DatasetUsage


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


def generate_experiment_dir(dataset_name, dataset_usage: DatasetUsage, name):
    return f"{dataset_name}/{dataset_usage.name}/{name}"


def generate_run_name():
    return datetime.now().strftime("%m-%d-%Y_%H-%M-%S")


def save_experiment_results(results, experiment_dir):
    # TODO: Make this not horrible
    print(results)
    stacked_seeds = [
        {
            ('archs',): [arch] * len(seeds),
            ('seeds',): list(seeds.keys()),
            **{
                (metric, measure): [run[metric][measure] for run in seeds.values()]
                for metric in list(seeds.values())[0].keys() for measure in list(list(seeds.values())[0].values())[0].keys()
            }
        }
        for arch, seeds in results.items()
    ]

    reformed_data = {
        key: [item for arch_results in stacked_seeds for item in arch_results[key]]
        for key in stacked_seeds[0].keys()
    }
    df = pd.DataFrame(reformed_data)
    df.to_csv(experiment_dir / 'results.csv', sep=';')  # Seperator other than comma due to architecture representation


def save_run(result, architecture, params, run_dir):
    _save_run_result(result, run_dir)
    _save_architecture(architecture, run_dir)
    _save_hyper_parameters(params, run_dir)


def _save_run_result(result, run_dir):
    filepath = run_dir / 'results.json'
    with open(filepath, 'w') as out:
        json.dump(result, out)


def _save_hyper_parameters(parameters, run_dir):
    filepath = run_dir / 'parameters.pkl'
    with open(filepath, 'wb') as out:
        pickle.dump(parameters, out)


def _save_architecture(architecture, run_dir):
    filepath = run_dir / 'architecture.pkl'
    with open(filepath, 'wb') as out:
        pickle.dump(architecture, out)


def load_architecture(run_dir):
    filepath = run_dir / 'architecture.pkl'
    with open(filepath, 'rb') as file:
        return pickle.load(file)
