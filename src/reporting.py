import json
import logging
import pickle
from datetime import datetime

import pandas as pd

from src.data import DatasetUsage


def generate_experiment_dir(dataset_name, dataset_usage: DatasetUsage, name):
    return f"{dataset_name}/{dataset_usage.name}/{name}"


def generate_run_name():
    return datetime.now().strftime("%m-%d-%Y_%H-%M-%S")


def save_experiment_results(results, experiment_dir):
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
    logging.info("Experiment results: \n" + df.to_string())
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
