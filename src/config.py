"""Define parameters to configure the pipeline.

Copyright (c) 2023, Harry Knighton
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

import json
import logging
from pathlib import Path

ROOT_DIR = Path(__file__).absolute().parent.parent
SRC_DIR = ROOT_DIR / 'src'
DATASETS_DIR = ROOT_DIR / 'datasets'
LOG_DIR = ROOT_DIR / 'logs'
EXPERIMENTS_DIR = ROOT_DIR / 'experiments'
REPORTS_DIR = ROOT_DIR / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

AUTOML_LOGGER = logging.getLogger("automl")
AUTOML_LOGGER.setLevel(logging.INFO)
_CONSOLE_HANDLER = logging.StreamHandler()
AUTOML_LOGGER.addHandler(_CONSOLE_HANDLER)

# Seeds used in https://chemrxiv.org/engage/chemrxiv/article-details/636fa49b80c9bfb4dc944c1c
# From https://github.com/davidbuterez/mf-pcba
with open(SRC_DIR / 'mf_pcba_seeds.json') as json_data:
    MF_PCBA_SEEDS = json.load(json_data)
    MF_PCBA_SEEDS = {'AID' + k: v for k, v in MF_PCBA_SEEDS.items()}

# Parameters used in "Multi-fidelity machine learning models for improved high-throughput screening predictions"
DEFAULT_LR = 1e-4
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR_PLATEAU_FACTOR = 0.5
DEFAULT_LR_PLATEAU_PATIENCE = 10
DEFAULT_LABEL_SCALER = 'standard'
DEFAULT_EARLY_STOP_PATIENCE = 30
DEFAULT_EARLY_STOP_DELTA = 0.
DEFAULT_TEST_SPLIT = 0.1
DEFAULT_TRAIN_VAL_SPLIT = 0.9

DEFAULT_SAVE_TRIALS_EVERY = 50
DEFAULT_PRECISION = 'medium'

# Number of data samples to use for proxy calculation
DEFAULT_PROXY_BATCH_SIZE = 64

MIN_SEED = 0
MAX_SEED = 1e6
