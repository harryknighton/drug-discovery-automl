from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn import model_selection
from torch import Tensor
from torch_geometric.data import Data, InMemoryDataset

from src.config import DATAFILE_NAME, RANDOM_SEEDS, DATA_DIR, DEFAULT_LOGGER
from src.metrics import StandardScaler
from src.parameters import DatasetUsage, HyperParameters, MFPCBA, BasicSplit, KFolds

_MAX_ATOMIC_NUM = 80
_N_FEATURES = _MAX_ATOMIC_NUM + 33


class HTSDataset(InMemoryDataset):
    def __init__(self, name: str, dataset_usage: DatasetUsage):
        self.name = name
        self.dataset_usage = dataset_usage
        self.sd_or_dr = 'SD' if self.dataset_usage == DatasetUsage.SDOnly else 'DR'
        root = str(DATA_DIR / name)
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.scaler = StandardScaler()
        self._scale_labels()

    def __get__(self, idx):
        return self.get(idx)

    @property
    def raw_file_names(self):
        return [DATAFILE_NAME]

    @property
    def processed_file_names(self):
        return [f'processed_{self.sd_or_dr.lower()}_data.pt']

    def process(self):
        DEFAULT_LOGGER.debug(f"Processing dataset {self.name} at {self.root}")
        df = _read_data(Path(self.root) / self.raw_file_names[0])
        if self.sd_or_dr == 'DR':
            df = df[df['DR'].notnull()]
        data_list = _process_data(df, self.sd_or_dr)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _scale_labels(self):
        DEFAULT_LOGGER.debug(f"Scaling dataset with scaler {self.scaler}")
        scaled_labels = self.scaler.fit_transform(self.data.y.reshape(-1, 1))
        self.data.y = torch.stack((scaled_labels.flatten(), self.data.y), dim=1)


def _read_data(filepath: Path) -> pd.DataFrame:
    return pd.read_csv(filepath)


def _process_data(df: pd.DataFrame, sd_or_dr) -> List[Data]:
    import chemprop
    _set_atomic_num(_MAX_ATOMIC_NUM)
    smiles = df['neut-smiles']
    mols = [chemprop.features.featurization.MolGraph(s) for s in smiles]
    xs = [Tensor(m.f_atoms) for m in mols]
    conns = [_get_connectivity(m) for m in mols]
    edge_indexes = [torch.tensor(conn, dtype=torch.long).T.contiguous() for conn in conns]
    ys = Tensor(df[sd_or_dr].values)
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


def partition_dataset(dataset, params: HyperParameters):
    if isinstance(params.dataset_split, MFPCBA):
        for seed in RANDOM_SEEDS[dataset.name]:
            yield seed, mf_pcba_split(dataset, seed)
    elif isinstance(params.dataset_split, BasicSplit):
        np.random.seed(params.random_seed)
        test_dataset, training_dataset = split_dataset(dataset, params.dataset_split.test_split)
        train_dataset, val_dataset = split_dataset(training_dataset, params.dataset_split.train_val_split)
        yield 0, (train_dataset, val_dataset, test_dataset)
    elif isinstance(params.dataset_split, KFolds):
        np.random.seed(params.random_seed)
        test_dataset, training_dataset = split_dataset(dataset, params.dataset_split.test_split)
        for i, (train_dataset, val_dataset) in enumerate(k_folds(training_dataset, params.dataset_split.k)):
            yield i, (train_dataset, val_dataset, test_dataset)
    else:
        raise ValueError("Unsupported dataset splitting scheme " + str(params.dataset_split))


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


def augment_dataset_with_sd_readouts(dataset: HTSDataset, model: torch.nn.Module):
    features = model(dataset.data.x, dataset.data.edge_index, dataset.data.batch).detach()
    assert features.shape[1] == dataset.data.x.shape[1]
    dataset.data.x = torch.stack((dataset.data.x, features), dim=1)


def get_num_input_features(dataset_usage: DatasetUsage):
    return _N_FEATURES + (1 if dataset_usage == DatasetUsage.DRWithSDReadouts else 0)
