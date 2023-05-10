"""Load and process high-throughput screening datasets.

Note: Raw data must be generated using https://github.com/davidbuterez/mf-pcba and placed in the datasets directory
prior to using this code to load the dataset.

References:
    David Buterez, Jon Paul Janet, Steven J. Kiddle, and Pietro Liò
    Journal of Chemical Information and Modeling 2023 63 (9), 2667-2678
    DOI: 10.1021/acs.jcim.2c01569

Copyright (c) 2023, Harry Knighton
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

from enum import auto, Enum
from pathlib import Path
from typing import List, Any

import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import Data, InMemoryDataset

from src.config import AUTOML_LOGGER

_MAX_ATOMIC_NUM = 80
_N_FEATURES = _MAX_ATOMIC_NUM + 33


class DatasetUsage(Enum):
    """Define ways in which the Dose Response (DR) and Single Dose (SD) data modalities can be used."""
    SDOnly = auto()
    DROnly = auto()
    DRWithSDLabels = auto()
    DRWithSDReadouts = auto()


def requires_sd_data(usage: DatasetUsage) -> bool:
    """Define which data usages require the Single Dose (SD) data to be loaded"""
    return (
        usage == DatasetUsage.SDOnly or
        usage == DatasetUsage.DRWithSDLabels
    )


def requires_dr_data(usage: DatasetUsage) -> bool:
    """Define which data usages require the Dose Response (DR) data to be loaded"""
    return (
        usage == DatasetUsage.DROnly or
        usage == DatasetUsage.DRWithSDLabels or
        usage == DatasetUsage.DRWithSDReadouts
    )


class HTSDataset(InMemoryDataset):
    """Load an MF-PCBA bioassay dataset and process it into a form suitable for machine learning.

    Note: Raw data must be generated using https://github.com/davidbuterez/mf-pcba and placed in the datasets directory
    prior to using this code to load the dataset.
    """
    def __init__(self, root: Path, dataset_usage: DatasetUsage, *args: Any, **kwargs: Any):
        self.dataset_usage = dataset_usage
        self.label_column = _get_label_column(dataset_usage)
        super().__init__(str(root), *args, **kwargs)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def __get__(self, idx):
        return self.get(idx)

    @property
    def num_classes(self) -> int:
        return 1  # Always only one regression target

    @property
    def raw_file_names(self):
        """Define filenames of raw data files."""
        if requires_dr_data(self.dataset_usage) and (Path(self.raw_dir) / 'DR.csv').exists():
            return ['DR.csv']  # DR data from separate file
        else:
            return ['SD.csv']  # DR data included in SD CSV file

    @property
    def processed_file_names(self):
        """Define files in which processed data is saved.

        Note: A file is created per dataset usage to speed-up loading.
        """
        name = ''
        if requires_dr_data(self.dataset_usage):
            name += 'dr'
        if requires_sd_data(self.dataset_usage):
            if name:
                name += '_'
            name += 'sd'
        return [f'processed_{name}_data.pt']

    def download(self):
        """Data must be present locally and will not be downloaded."""
        pass

    def process(self):
        """Load the raw data file and process it into a tensor representation."""
        AUTOML_LOGGER.debug(f"Processing dataset at {self.root}")
        df = _read_data(Path(self.raw_paths[0]))
        if requires_sd_data(self.dataset_usage):
            df = df[df['SD'].notnull()]
        if requires_dr_data(self.dataset_usage):
            df = df[df['DR'].notnull()]
        data_list = _process_data(
            df,
            label_col=self.label_column,
            include_sd_labels=self.dataset_usage == DatasetUsage.DRWithSDLabels
        )
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def augment_dataset_with_sd_readouts(self, model: torch.nn.Module):
        """When using DatasetUsage.DRWithSDReadouts, predict SD labels using `model` and add them to the dataset.

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
