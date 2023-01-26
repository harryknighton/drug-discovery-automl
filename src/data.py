import random
from typing import List

import chemprop
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch import Tensor
from torch_geometric.data import Data, InMemoryDataset

from src.config import DATAFILE_NAME


class HTSDataset(InMemoryDataset):
    def __init__(self, root, sd_or_dr):
        self.sd_or_dr = sd_or_dr
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
        df = _read_data(self.root / self.raw_file_names[0])
        if self.sd_or_dr == 'DR':
            df = df[df['DR'].notnull()]
        data_list = _process_data(df, self.sd_or_dr)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def _read_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def _process_data(df: pd.DataFrame, sd_or_dr) -> List[Data]:
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


def split_dataset(dataset, ratio):
    split = int(ratio * len(dataset))
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    # TODO: Randomly split dataset
    training_dataset = dataset.index_select(indices[:split])
    validation_dataset = dataset.index_select(indices[split:])
    return training_dataset, validation_dataset


def k_folds(dataset, k, seed):
    kfold = KFold(n_splits=k, shuffle=True, random_state=seed)
    for train_index, val_index in kfold.split(dataset):
        train_dataset = dataset.index_select(train_index.tolist())
        val_dataset = dataset.index_select(val_index.tolist())
        yield train_dataset, val_dataset
