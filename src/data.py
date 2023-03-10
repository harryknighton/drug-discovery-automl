from abc import ABC
from dataclasses import dataclass
from enum import auto, Enum
from pathlib import Path
from typing import List, Any, Type, Optional

import numpy as np
import pandas as pd
import torch
import torch_geometric
from sklearn import model_selection
from torch import Tensor
from torch_geometric.data import Data, InMemoryDataset, Dataset

from src.config import DATAFILE_NAME, DATA_DIR, DEFAULT_LOGGER, MF_PCBA_SEEDS

_MAX_ATOMIC_NUM = 80
_N_FEATURES = _MAX_ATOMIC_NUM + 33


class DatasetUsage(Enum):
    SDOnly = auto()
    DROnly = auto()
    DRWithSDLabels = auto()
    DRWithSDReadouts = auto()


class DatasetSplit(ABC):
    pass


@dataclass(frozen=True)
class KFolds(DatasetSplit):
    k: int
    test_split: float


@dataclass(frozen=True)
class MFPCBA(DatasetSplit):
    seeds: List[int]


@dataclass(frozen=True)
class BasicSplit(DatasetSplit):
    test_split: float
    train_val_split: float


class HTSDataset(InMemoryDataset):
    def __init__(self, root: Path, dataset_usage: DatasetUsage, *args: Any, **kwargs: Any):
        self.dataset_usage = dataset_usage
        self.label_column = _get_label_column(dataset_usage)
        super().__init__(str(root), *args, **kwargs)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def __get__(self, idx):
        return self.get(idx)

    @property
    def num_classes(self) -> int:
        return 1

    @property
    def raw_file_names(self):
        return [DATAFILE_NAME]

    @property
    def processed_file_names(self):
        match self.dataset_usage:
            case DatasetUsage.DROnly | DatasetUsage.DRWithSDReadouts: name = 'dr'
            case DatasetUsage.SDOnly: name = 'sd'
            case DatasetUsage.DRWithSDLabels: name = 'dr_sd'
            case _: raise ValueError("Unsupported DatasetUsage")
        return [f'processed_{name}_data.pt']

    def download(self):
        pass

    def process(self):
        DEFAULT_LOGGER.debug(f"Processing dataset at {self.root}")
        df = _read_data(Path(self.raw_paths[0]))
        df = df[df[self.label_column].notnull()]
        data_list = _process_data(
            df,
            label_col=self.label_column,
            include_sd_labels=self.dataset_usage == DatasetUsage.DRWithSDLabels
        )
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def augment_dataset_with_sd_readouts(self, model: torch.nn.Module):
        """Predict SD labels using `model` and add them to the dataset

        WARNING: This must be called before the dataset is accessed externally as when `self.get()` is called
            the current `self.data.x` is cached in `self._data_list`
        """
        assert self._data_list is None or all(x is None for x in self._data_list)
        x, edge_index = self.data.x, self.data.edge_index
        batch_indices = torch.arange(len(self.data.y), dtype=torch.int64)
        slice_widths = torch.diff(self.slices['x'])
        batch = torch.repeat_interleave(batch_indices, slice_widths)
        features = model(x, edge_index, batch).detach()
        expanded_sd_labels = features[batch]
        self.data.x = torch.cat((self.data.x, expanded_sd_labels), dim=1)


def _get_label_column(dataset_usage: DatasetUsage) -> str:
    match dataset_usage:
        case DatasetUsage.SDOnly:
            return 'SD'
        case DatasetUsage.DROnly | DatasetUsage.DRWithSDReadouts | DatasetUsage.DRWithSDLabels:
            return 'DR'
        case _:
            raise ValueError("Unsupported DatasetUsage")


def _read_data(filepath: Path) -> pd.DataFrame:
    return pd.read_csv(filepath)


def _process_data(df: pd.DataFrame, label_col: str, include_sd_labels: bool) -> List[Data]:
    import chemprop
    _set_atomic_num(_MAX_ATOMIC_NUM)
    smiles = df['neut-smiles']
    mols = [chemprop.features.featurization.MolGraph(s) for s in smiles]
    xs = [Tensor(m.f_atoms) for m in mols]
    if include_sd_labels:
        assert len(xs) == len(df['SD'])
        xs = [torch.cat((x, torch.full((x.shape[0], 1), label)), dim=1) for x, label in zip(xs, df['SD'])]
    conns = [_get_connectivity(m) for m in mols]
    edge_indexes = [torch.tensor(conn, dtype=torch.long).T.contiguous() for conn in conns]
    ys = Tensor(df[label_col].values)
    return [Data(x=x, edge_index=index, y=y) for x, index, y in zip(xs, edge_indexes, ys)]


def _get_connectivity(mol):
    connections = []
    b2a = mol.b2a
    a2b = mol.a2b
    for aI, bonds in enumerate(a2b):
        neighbours = [(b2a[bI], aI) for bI in bonds]
        connections.extend(neighbours)
    return connections


def _set_atomic_num(num):
    """Set the number of features used when one-hot encoding atomic numbers"""
    from chemprop.features.featurization import PARAMS
    PARAMS.MAX_ATOMIC_NUM = num
    PARAMS.ATOM_FEATURES['atomic_num'] = list(range(num))
    PARAMS.ATOM_FDIM = sum(len(choices) + 1 for choices in PARAMS.ATOM_FEATURES.values()) + 2


def get_dataset(dataset_name: str, dataset_usage: Optional[DatasetUsage] = None, **kwargs: Any) -> Dataset:
    root = DATA_DIR / dataset_name
    if dataset_name.startswith('AID'):
        return HTSDataset(root, dataset_usage=dataset_usage, **kwargs)
    elif dataset_name == 'QM7b':
        return torch_geometric.datasets.QM7b(root, **kwargs)
    elif dataset_name == 'QM9':
        return torch_geometric.datasets.QM9(root, **kwargs)
    else:
        raise ValueError(f"Dataset {dataset_name} not recognised")


def partition_dataset(dataset: Dataset, dataset_split: DatasetSplit, random_seed: Optional[int] = 0):
    if isinstance(dataset_split, MFPCBA):
        for seed in MF_PCBA_SEEDS[dataset.name]:
            yield seed, mf_pcba_split(dataset, seed)
    elif isinstance(dataset_split, BasicSplit):
        np.random.seed(random_seed)
        test_dataset, training_dataset = split_dataset(dataset, dataset_split.test_split)
        train_dataset, val_dataset = split_dataset(training_dataset, dataset_split.train_val_split)
        yield 0, (train_dataset, val_dataset, test_dataset)
    elif isinstance(dataset_split, KFolds):
        np.random.seed(random_seed)
        test_dataset, training_dataset = split_dataset(dataset, dataset_split.test_split)
        for i, (train_dataset, val_dataset) in enumerate(k_folds(training_dataset, dataset_split.k)):
            yield i, (train_dataset, val_dataset, test_dataset)
    else:
        raise ValueError("Unsupported dataset splitting scheme " + str(dataset_split))


def mf_pcba_split(dataset: HTSDataset, seed: int):
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


def split_dataset(dataset: HTSDataset, ratio: float):
    split = int(ratio * len(dataset))
    indices = np.random.permutation(np.arange(len(dataset), dtype=np.int64))
    training_dataset = dataset.index_select(indices[:split])
    validation_dataset = dataset.index_select(indices[split:])
    return training_dataset, validation_dataset


def k_folds(dataset: HTSDataset, k: int):
    kfold = model_selection.KFold(n_splits=k)
    for train_index, val_index in kfold.split(dataset):
        train_dataset = dataset.index_select(train_index.tolist())
        val_dataset = dataset.index_select(val_index.tolist())
        yield train_dataset, val_dataset


class Scaler(torch.nn.Module, ABC):
    def fit(self, values: Tensor) -> Tensor:
        raise NotImplementedError()

    def transform(self, values: Tensor) -> Tensor:
        raise NotImplementedError()

    def inverse_transform(self, values: Tensor) -> Tensor:
        raise NotImplementedError()

    def fit_transform(self, values: Tensor) -> Tensor:
        self.fit(values)
        return self.transform(values)

    @staticmethod
    def _validate_input(values) -> None:
        if values.dim() != 2:
            raise ValueError("Values must be shape (n_samples, n_features) but got " + str(values.shape))


class StandardScaler(Scaler):
    def __init__(self, epsilon: float = 1e-7):
        """Adapted from https://gist.github.com/farahmand-m/8a416f33a27d73a149f92ce4708beb40"""
        super().__init__()
        self.register_buffer('means', Tensor(), persistent=False)
        self.register_buffer('stds', Tensor(), persistent=False)
        self.register_buffer('epsilon', torch.tensor([epsilon]), persistent=False)

    def fit(self, values: Tensor):
        self._validate_input(values)
        if values.shape[0] <= 1:
            raise ValueError("Must have more than one sample to fit to, got " + str(values.shape[0]))
        self.means = torch.mean(values, dim=0)
        self.stds = torch.std(values, dim=0)
        assert len(self.means) == len(self.stds) == values.shape[-1]

    def transform(self, values: Tensor):
        self._validate_input(values)
        return (values - self.means) / (self.stds + self.epsilon)

    def inverse_transform(self, values):
        self._validate_input(values)
        return values * (self.stds + self.epsilon) + self.means


class MinMaxScaler(Scaler):
    def __init__(self, scaled_min: int = 0, scaled_max: int = 1, epsilon: float = 1e-7):
        super().__init__()
        self.register_buffer('scaled_min', torch.tensor([scaled_min]), persistent=False)
        self.register_buffer('scaled_max', torch.tensor([scaled_max]), persistent=False)
        self.register_buffer('mins', Tensor(), persistent=False)
        self.register_buffer('maxs', Tensor(), persistent=False)
        self.register_buffer('epsilon', torch.tensor([epsilon]), persistent=False)

    def fit(self, values: Tensor):
        self._validate_input(values)
        if values.shape[0] <= 1:
            raise ValueError("Must have more than one sample to fit to, got " + str(values.shape[0]))
        self.mins = torch.min(values, dim=0).values
        self.maxs = torch.max(values, dim=0).values
        assert len(self.mins) == len(self.maxs) == values.shape[-1]

    def transform(self, values: Tensor):
        self._validate_input(values)
        standard_scale = (values - self.mins) / (self.maxs - self.mins + self.epsilon)
        return standard_scale * (self.scaled_max - self.scaled_min) + self.scaled_min

    def inverse_transform(self, values):
        self._validate_input(values)
        standard_scale = (values - self.scaled_min) / (self.scaled_max - self.scaled_min)
        return standard_scale * (self.maxs - self.mins + self.epsilon) + self.mins


class NamedLabelledDataset:
    def __init__(self, name: str, dataset: Dataset, label_scaler: Scaler):
        self.name = name
        self.dataset = dataset
        self.label_scaler = label_scaler


def fit_label_scaler(dataset: Dataset, scaler_type: Type[Scaler]) -> Scaler:
    scaler = scaler_type()
    scaler.fit(dataset.data.y.reshape(-1, 1))
    return scaler
