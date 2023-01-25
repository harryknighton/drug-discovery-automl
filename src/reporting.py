import json
import pickle
from datetime import datetime
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torchmetrics import PearsonCorrCoef


class PearsonCorrCoefSquared(PearsonCorrCoef):
    def compute(self) -> Tensor:
        r = super(PearsonCorrCoefSquared, self).compute()
        return torch.pow(r, 2)


def generate_experiment_path(dataset_name, using_embeddings):
    data_used = 'SD_DR' if using_embeddings else 'DR'
    return f"{dataset_name}/{data_used}"


def generate_run_name():
    return datetime.now().strftime("%m-%d-%Y_%H-%M-%S")


def save_hyper_parameters(parameters, architecture, run_dir):
    filepath = run_dir + '/' + 'parameters.txt'
    with open(filepath, 'w') as out:
        out.write(str(architecture))
        out.write('\n')
        out.write(str(parameters))


def save_architecture(architecture, run_dir):
    filepath = run_dir + '/' + 'architecture.pkl'
    with open(filepath, 'wb') as out:
        pickle.dump(architecture, out)


def load_architecture(run_dir):
    filepath = run_dir + '/' + 'architecture.pkl'
    with open(filepath, 'rb') as file:
        return pickle.load(file)


def save_k_fold_metrics(fold_metrics, run_dir):
    filepath = run_dir + '/' + 'results.json'
    stacked_metrics = {name: np.array([fold[name] for fold in fold_metrics]) for name in fold_metrics[0]}
    means = {name: float(np.mean(metrics)) for name, metrics in stacked_metrics.items()}
    variances = {name: float(np.var(metrics)) for name, metrics in stacked_metrics.items()}
    metrics = {name: {'mean': means[name], 'variance': variances[name]} for name in stacked_metrics}
    with open(filepath, 'w') as out:
        json.dump(metrics, out)
