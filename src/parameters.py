from abc import ABC
from dataclasses import dataclass
from enum import auto, Enum
from typing import List


class DatasetUsage(Enum):
    SDOnly = auto()
    DROnly = auto()
    DRWithSDReadouts = auto()
    DRWithSDEmbeddings = auto()


class DatasetSplit(ABC):
    pass


@dataclass(frozen=True)
class KFolds(DatasetSplit):
    k: int


@dataclass(frozen=True)
class MFPCBA(DatasetSplit):
    seeds: List[int]


class BasicSplit(DatasetSplit):
    pass


@dataclass
class HyperParameters:
    random_seed: int
    use_sd_readouts: bool
    dataset_split: DatasetSplit
    test_split: float
    train_val_split: float
    batch_size: int
    early_stop_patience: int
    early_stop_min_delta: float
    lr: float
    max_epochs: int
    num_workers: int
