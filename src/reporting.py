import json
import pickle
from datetime import datetime

import pandas as pd
import torch
from torch import Tensor
from torchmetrics import PearsonCorrCoef, MeanSquaredError


class PearsonCorrCoefSquared(PearsonCorrCoef):
    """Provides an alternative implementation of R^2"""
    def compute(self) -> Tensor:
        r = super(PearsonCorrCoefSquared, self).compute()
        return torch.pow(r, 2)


class RootMeanSquaredError(MeanSquaredError):
    def __init__(self):
        super(RootMeanSquaredError, self).__init__(squared=False)


def generate_experiment_dir(dataset_name, using_sd_readouts, name):
    data_used = 'SD_DR' if using_sd_readouts else 'DR'
    return f"{dataset_name}/{data_used}/{name}"


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
