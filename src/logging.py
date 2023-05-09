"""Provide utilities to analyse and log results from the pipeline.

Copyright (c) 2023, Harry Knighton
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import LOG_DIR, AUTOML_LOGGER
from src.data.hts import HTSDataset
from src.data.utils import NamedLabelledDataset
from src.types import Metrics


def generate_experiment_dir(dataset: NamedLabelledDataset, experiment_name: str) -> Path:
    experiment_dir = LOG_DIR / dataset.name
    if isinstance(dataset.dataset, HTSDataset):
        experiment_dir /= dataset.dataset.dataset_usage.name
    experiment_dir /= experiment_name
    return experiment_dir


def generate_run_name() -> str:
    return datetime.now().strftime("%m-%d-%Y_%H-%M-%S")


def save_experiment_results(results, experiment_dir: Path, filename: str) -> None:
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
    AUTOML_LOGGER.info(f"Experiment results: \n{architectures.to_string()}\n{logging_values.to_string()}")
    df.to_csv(experiment_dir / (filename + '.csv'), sep=';')  # Comma separator used in architecture representation


def save_run_results(results: dict[str, Metrics], run_dir: Path, filename: str) -> None:
    filepath = run_dir / (filename + '.json')
    trial_results = {version: {k: float(v) for k, v in trial.items()} for version, trial in results.items()}
    with open(filepath, 'w') as out:
        json.dump(trial_results, out, indent=2)


def read_json_file(parent_dir: Path, filename: str) -> None:
    filepath = parent_dir / filename
    with open(filepath, 'rb') as file:
        return json.load(file)


def analyse_results_distribution(results: dict[str | int, Metrics]) -> dict[str, Metrics]:
    """Calculate the distribution of results over all random seeds and data splits for a run"""
    assert len(results) > 0
    run_metrics = list(results.values())
    stacked_metrics = {
        metric: np.array([float(metrics[metric]) for metrics in run_metrics])
        for metric in run_metrics[0]
    }
    metrics = {}
    for metric, values in stacked_metrics.items():
        percentiles = np.percentile(values, [0, 25, 50, 75, 100])
        variance = np.var(values, ddof=1) if len(values) > 1 else 0.0
        metrics[metric] = {
            'mean': np.mean(values),
            'variance': variance,
            'min': percentiles[0],
            'p25': percentiles[1],
            'median': percentiles[2],
            'p75': percentiles[3],
            'max': percentiles[4]
        }
    return metrics
