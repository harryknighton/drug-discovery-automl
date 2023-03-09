import json
import pickle
from collections import defaultdict
from datetime import datetime

import pandas as pd

from src.config import DEFAULT_LOGGER
from src.data import NamedLabelledDataset, HTSDataset


def generate_experiment_dir(dataset: NamedLabelledDataset, experiment_name: str):
    if isinstance(dataset.dataset, HTSDataset):
        return dataset.name + '/' + dataset.dataset.dataset_usage.name + '/' + experiment_name
    else:
        return dataset.name + '/' + experiment_name


def generate_run_name():
    return datetime.now().strftime("%m-%d-%Y_%H-%M-%S")


def save_experiment_results(results, experiment_dir):
    reformed_results = defaultdict(list)
    for architecture, metrics in results.items():
        reformed_results[('architectures',)].append(architecture)
        for metric, measures in metrics.items():
            for measure, value in measures.items():
                reformed_results[(metric, measure)].append(value)
    df = pd.DataFrame(reformed_results)
    architectures = df[['architectures']]
    measures = df.columns.get_level_values(1)
    logging_values = df.iloc[:, (measures == 'mean') | (measures == 'variance')]
    DEFAULT_LOGGER.info(f"Experiment results: \n{architectures.to_string()}\n{logging_values.to_string()}")
    df.to_csv(experiment_dir / 'results.csv', sep=';')  # Seperator other than comma due to architecture representation


def save_run(trial_results, architecture, params, run_dir):
    _save_trial_results(trial_results, run_dir)
    _save_architecture(architecture, run_dir)
    _save_hyper_parameters(params, run_dir)


def _save_trial_results(results, run_dir):
    filepath = run_dir / 'results.json'
    trial_results = {version: {k: float(v) for k, v in trial.items()} for version, trial in results.items()}
    with open(filepath, 'w') as out:
        json.dump(trial_results, out, indent=2)


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
