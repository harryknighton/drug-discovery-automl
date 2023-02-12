from pathlib import Path

DATAFILE_NAME = 'SD.csv'

ROOT_DIR = Path(__file__).absolute().parent.parent
DATA_DIR = ROOT_DIR / 'data'
LOG_DIR = ROOT_DIR / 'logs'

DEFAULT_N_FEATURES = 113


# Seeds used in https://chemrxiv.org/engage/chemrxiv/article-details/636fa49b80c9bfb4dc944c1c
# From https://github.com/davidbuterez/mf-pcba
RANDOM_SEEDS = {
    "AID1445": [946067, 721263, 691383, 374914, 724299],
    "AID504329": [966204, 681725, 635271, 220018, 548422],
    "AID624330": [693665, 109746, 780835, 662995, 865845]
}
