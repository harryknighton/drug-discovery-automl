# Automated Machine Learning for Drug Discovery
This repository contains the code for my third-year university project. It can be used to run machine learning
experiments and network architecture search on the task of compound bioactivity prediction.

## Installation
Required dependencies are listed in requirements.txt.

## Using the Repository
### Loading datasets
Raw data files must be obtained through https://github.com/davidbuterez/mf-pcba and placed in the [datasets](datasets)
directory.
Processing is performed automatically the first time that a dataset is loaded using the [HTSDataset](src/data/hts.py)
class.

### Defining an experiment
Experiments are defined and stored in the [experiments](experiments) directory.
There are two examples that demonstrate how to configure an experiment:
[experiment_template.toml](experiments/experiment_template.toml) and
[nas_template.toml](experiments/nas_template.toml).

### Running the pipeline
The pipeline can be run through the (run_experiment.py)[scripts/run_experiment.py] file in the (scripts)[scripts]
directory. For example, it can be invoked as
```commandline
    python -m scripts.run_experiment --experiment-name baseline --dataset AID1445 --dataset-usage DROnly --num-workers 32
```
Slurm scripts are provided to use the repository on HPC services.

## Citing
This work is planned to be published in the future. There is currently no available citation.

## Licensing
Copyright (c) 2023, Harry Knighton
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
