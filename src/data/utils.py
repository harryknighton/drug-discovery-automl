"""Supply utility functions to load, handle and split datasets.

Copyright (c) 2023, Harry Knighton
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

from abc import ABC
from dataclasses import dataclass
from typing import List, Any, Optional

import numpy as np
import torch
import torch_geometric
from sklearn import model_selection
from torch_geometric.data import Dataset

from src.config import DATASETS_DIR, MF_PCBA_SEEDS
from src.data.hts import HTSDataset
from src.data.scaling import Scaler


class NamedLabelledDataset:
    """Wrap a dataset object with a name and a scaler to scale the dataset labels."""
    def __init__(self, name: str, dataset: Dataset, label_scaler: Scaler):
        self.name = name
        self.dataset = dataset
        self.label_scaler = label_scaler

    def to(self, device: torch.device | str) -> None:
        self.label_scaler = self.label_scaler.to(device)
        self.dataset.data = self.dataset.data.to(device)


class DatasetSplit(ABC):
    """Base class to define methods of splitting datasets./"""
    pass


@dataclass(frozen=True)
class KFolds(DatasetSplit):
    """Split a dataset into `k` training and validation folds with `test_split` left out"""
    k: int
    test_split: float  # Proportion of data to use in training


@dataclass(frozen=True)
class MFPCBA(DatasetSplit):
    """Use the data splitting methodology used in MF-PCBA"""
    seeds: List[int]


@dataclass(frozen=True)
class BasicSplit(DatasetSplit):
    """Split a dataset into training, validation and testing datasets"""
    test_split: float
    train_val_split: float


def get_dataset(dataset_name: str, **kwargs: Any) -> Dataset:
    """Get the dataset called `dataset_name`."""
    root = DATASETS_DIR / dataset_name
    if dataset_name.startswith('AID'):
        if 'dataset_usage' not in kwargs:
            raise ValueError("Must provide DatasetUsage for HTSDataset")
        return HTSDataset(root, dataset_usage=kwargs['dataset_usage'])
    elif dataset_name == 'QM7b':
        return torch_geometric.datasets.QM7b(root)
    elif dataset_name == 'QM9':
        return torch_geometric.datasets.QM9(root)
    else:
        raise ValueError(f"Dataset {dataset_name} not recognised")


def partition_dataset(dataset: NamedLabelledDataset, dataset_split: DatasetSplit, random_seed: Optional[int] = 0):
    """Split `dataset` using the methodology specified by `dataset_split`."""
    if isinstance(dataset_split, MFPCBA):
        for seed in MF_PCBA_SEEDS[dataset.name]:
            yield seed, mf_pcba_split(dataset.dataset, seed)
    elif isinstance(dataset_split, BasicSplit):
        np.random.seed(random_seed)
        test_dataset, training_dataset = split_dataset(dataset.dataset, dataset_split.test_split)
        train_dataset, val_dataset = split_dataset(training_dataset, dataset_split.train_val_split)
        yield 0, (train_dataset, val_dataset, test_dataset)
    elif isinstance(dataset_split, KFolds):
        np.random.seed(random_seed)
        test_dataset, training_dataset = split_dataset(dataset.dataset, dataset_split.test_split)
        for i, (train_dataset, val_dataset) in enumerate(k_folds(training_dataset, dataset_split.k)):
            yield i, (train_dataset, val_dataset, test_dataset)
    else:
        raise ValueError("Unsupported dataset splitting scheme " + str(dataset_split))


def mf_pcba_split(dataset: Dataset, seed: int):
    """Implement MF-PCBA splitting method."""
    # Splits used in https://github.com/davidbuterez/mf-pcba/blob/main/split_DR_with_random_seeds.ipynb
    # Adapted from https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test
    np.random.seed(seed)
    size = len(dataset)
    perm = np.random.permutation(np.arange(size, dtype=np.int64))
    train_end = int(0.8 * size)
    validate_end = int(0.1 * size) + train_end
    train = dataset.index_select(perm[:train_end])
    validate = dataset.index_select(perm[train_end:validate_end])
    test = dataset.index_select(perm[validate_end:])
    return train, validate, test


def split_dataset(dataset: Dataset, ratio: float):
    """Implement `BasicSplit` splitting method."""
    split = int(ratio * len(dataset))
    indices = np.random.permutation(np.arange(len(dataset), dtype=np.int64))
    training_dataset = dataset.index_select(indices[:split])
    validation_dataset = dataset.index_select(indices[split:])
    return training_dataset, validation_dataset


def k_folds(dataset: Dataset, k: int):
    """Implement `KFold` splitting method."""
    kfold = model_selection.KFold(n_splits=k)
    for train_index, val_index in kfold.split(dataset):
        train_dataset = dataset.index_select(train_index.tolist())
        val_dataset = dataset.index_select(val_index.tolist())
        yield train_dataset, val_dataset
