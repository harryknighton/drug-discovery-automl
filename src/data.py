import logging
import random
from enum import auto, Enum
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch import Tensor
from torch_geometric.data import Data, InMemoryDataset

from src.config import DATAFILE_NAME, RANDOM_SEEDS, DATA_DIR
from src.models import HyperParameters


class DatasetUsage(Enum):
    SDOnly = auto()
    DROnly = auto()
    DRWithSDReadouts = auto()
    DRWithSDEmbeddings = auto()


class HTSDataset(InMemoryDataset):
    def __init__(self, name: str, dataset_usage: DatasetUsage):
        self.name = name
        self.dataset_usage = dataset_usage
        self.sd_or_dr = 'SD' if self.dataset_usage == DatasetUsage.SDOnly else 'DR'
        root = str(DATA_DIR / name)
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def __get__(self, idx):
        return self.get(idx)

    @property
    def raw_file_names(self):
        return [DATAFILE_NAME]

    @property
    def processed_file_names(self):
        return [f'processed_{self.sd_or_dr.lower()}_data.pt']

    def process(self):
        df = _read_data(Path(self.root) / self.raw_file_names[0])
        if self.sd_or_dr == 'DR':
            df = df[df['DR'].notnull()]
        data_list = _process_data(df, self.sd_or_dr)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def _read_data(filepath: Path) -> pd.DataFrame:
    return pd.read_csv(filepath)


def _process_data(df: pd.DataFrame, sd_or_dr) -> List[Data]:
    import chemprop
    _set_atomic_num(80)
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


def partition_dataset(dataset, params: HyperParameters, use_mf_pcba_scheme: bool = False):
    if use_mf_pcba_scheme:
        for seed in RANDOM_SEEDS[dataset.name]:
            logging.info("Using MF_PCBA split with seed " + str(seed))
            yield mf_pcba_split(dataset, seed)
    else:
        np.random.seed(params.random_seed)
        test_dataset, training_dataset = split_dataset(dataset, params.test_split)
        if params.k_folds == 1:
            yield *split_dataset(dataset, params.train_val_split), test_dataset
        else:
            for train_dataset, val_dataset in k_folds(dataset, params.k_folds):
                yield train_dataset, val_dataset, test_dataset


def mf_pcba_split(dataset: HTSDataset, seed: int):
    # Splits used in https://github.com/davidbuterez/mf-pcba/blob/main/split_DR_with_random_seeds.ipynb
    # Adapted from https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test
    np.random.seed(seed)
    size = len(dataset)
    perm = np.random.permutation(size)
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
    kfold = KFold(n_splits=k)
    for train_index, val_index in kfold.split(dataset):
        train_dataset = dataset.index_select(train_index.tolist())
        val_dataset = dataset.index_select(val_index.tolist())
        yield train_dataset, val_dataset
