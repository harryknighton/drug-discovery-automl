import logging
from pathlib import Path

DEFAULT_LOGGER = logging.getLogger("automl")
DEFAULT_LOGGER.setLevel(logging.INFO)
_CONSOLE_HANDLER = logging.StreamHandler()
DEFAULT_LOGGER.addHandler(_CONSOLE_HANDLER)

DATAFILE_NAME = 'SD.csv'

ROOT_DIR = Path(__file__).absolute().parent.parent
DATASETS_DIR = ROOT_DIR / 'datasets'
LOG_DIR = ROOT_DIR / 'logs'
EXPERIMENTS_DIR = ROOT_DIR / 'experiments'


# Seeds used in https://chemrxiv.org/engage/chemrxiv/article-details/636fa49b80c9bfb4dc944c1c
# From https://github.com/davidbuterez/mf-pcba
MF_PCBA_SEEDS = {
    "AID1445": [946067, 721263, 691383, 374914, 724299],
    "AID504329": [966204, 681725, 635271, 220018, 548422],
    "AID624330": [693665, 109746, 780835, 662995, 865845]
}

# Parameters used in "Multi-fidelity machine learning models for improved high-throughput screening predictions"
DEFAULT_LR = 1e-4
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR_PLATEAU_FACTOR = 0.5
DEFAULT_LR_PLATEAU_PATIENCE = 10

# Default parameters determined by experimentation
DEFAULT_LABEL_SCALER = 'standard'
DEFAULT_EARLY_STOP_PATIENCE = 30
DEFAULT_EARLY_STOP_DELTA = 0.
DEFAULT_TEST_SPLIT = 0.1
DEFAULT_TRAIN_VAL_SPLIT = 0.9

DEFAULT_SAVE_TRIALS_EVERY = 100
DEFAULT_PRECISION = 'medium'
