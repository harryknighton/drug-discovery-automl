from pathlib import Path

from torchmetrics import MetricCollection, R2Score, MeanAbsoluteError, MeanSquaredError, PearsonCorrCoef

from src.reporting import PearsonCorrCoefSquared, RootMeanSquaredError
from src.models import HyperParameters

DATAFILE_NAME = 'SD.csv'

ROOT_DIR = Path(__file__).absolute().parent.parent
DATA_DIR = ROOT_DIR / 'data'
LOG_DIR = ROOT_DIR / 'logs'

NUM_WORKERS = 0

DEFAULT_N_FEATURES = 113

DEFAULT_PARAMETERS = HyperParameters(
    random_seed=1424,
    use_sd_readouts=False,
    k_folds=6,
    test_split=0.2,
    train_val_split=0.75,
    batch_size=32,
    early_stop_patience=10,
    early_stop_min_delta=0.01,
    lr=0.0001,
    max_epochs=150
)

DEFAULT_METRICS = MetricCollection([
    MeanAbsoluteError(),
    RootMeanSquaredError(),
    PearsonCorrCoef(),
    PearsonCorrCoefSquared(),
    R2Score()
])

# Seeds used in https://chemrxiv.org/engage/chemrxiv/article-details/636fa49b80c9bfb4dc944c1c
# From https://github.com/davidbuterez/mf-pcba
RANDOM_SEEDS = {
    "AID1445": [946067, 721263, 691383, 374914, 724299],
    "AID504329": [966204, 681725, 635271, 220018, 548422],
    "AID624330": [693665, 109746, 780835, 662995, 865845]
}
