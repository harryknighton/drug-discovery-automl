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
    train_val_split=0.8,
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
