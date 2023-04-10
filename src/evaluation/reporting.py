import json
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import AUTOML_LOGGER, LOG_DIR
from src.data.hts import HTSDataset
from src.data.utils import NamedLabelledDataset
from src.types import Metrics


def generate_experiment_dir(dataset: NamedLabelledDataset, experiment_name: str, version: Optional[int] = None) -> Path:
    experiment_dir = LOG_DIR / dataset.name
    if isinstance(dataset.dataset, HTSDataset):
        experiment_dir /= dataset.dataset.dataset_usage.name
    experiment_dir /= experiment_name
    if version is not None:
        return Path(f'{experiment_dir}_{version}')
    counter = 0
    version_dir = Path(str(experiment_dir) + '_0')
    while version_dir.exists():
        counter += 1
        version_dir = Path(str(experiment_dir) + '_' + str(counter))
    return version_dir


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


def load_architecture(run_dir) -> None:
    filepath = run_dir / 'architecture.pkl'
    with open(filepath, 'rb') as file:
        return pickle.load(file)