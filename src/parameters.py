from abc import ABC
from dataclasses import dataclass
from enum import auto, Enum
from typing import List, Type

from src.data import Scaler


class DatasetUsage(Enum):
    SDOnly = auto()
    DROnly = auto()
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


@dataclass
class HyperParameters:
    random_seed: int
    dataset_usage: DatasetUsage
    dataset_split: DatasetSplit
    label_scaler: Type[Scaler]
    limit_batches: float
    batch_size: int
    early_stop_patience: int
    early_stop_min_delta: float
    lr: float
    max_epochs: int
    num_workers: int
