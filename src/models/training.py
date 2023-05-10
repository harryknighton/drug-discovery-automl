"""Implement the training and testing loops to train and evaluate models.

Copyright (c) 2023, Harry Knighton
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Type, Tuple, Any

import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor
from torch.nn import MSELoss
from torch.optim import Adam
import pytorch_lightning as tl
from torch_geometric.data import LightningDataset
from torch_geometric.data.lightning_datamodule import LightningDataModule
from torchmetrics import MetricCollection

from src.config import DEFAULT_LR_PLATEAU_PATIENCE, DEFAULT_LR_PLATEAU_FACTOR, MAX_SEED, MIN_SEED, AUTOML_LOGGER
from src.data.scaling import Scaler
from src.data.utils import DatasetSplit, NamedLabelledDataset, partition_dataset
from src.models.metrics import DEFAULT_ACCURACY_METRICS, DEFAULT_EXPLAINABILITY_METRICS, detach_metrics
from src.models.gnns import GNNArchitecture, GNN
from src.nas.proxies import DEFAULT_PROXIES
from src.logging import (
    generate_run_name, save_experiment_results, save_run_results, analyse_results_distribution
)
from src.types import Metrics


@dataclass
class HyperParameters:
    """Define hyperparameters needed to train a model."""
    random_seeds: List[int]
    dataset_split: DatasetSplit
    label_scaler: Type[Scaler]
    batch_size: int
    early_stop_patience: int
    early_stop_min_delta: float
    lr: float
    max_epochs: int
    num_workers: int
    precision: str


class LitGNN(tl.LightningModule):
    """Define training and testing loop for a GNN model."""
    def __init__(
        self,
        model: GNN,
        params: HyperParameters,
        metrics: MetricCollection,
        explainability_metrics: MetricCollection,
        label_scaler: Scaler
    ) -> None:
        super().__init__()
        self.model = model
        self.params = params
        self.loss = MSELoss()
        self.val_metrics = metrics.clone()
        self.test_metrics = metrics.clone()
        self.test_results = None
        self.explainability_metrics = explainability_metrics.clone()
        self.label_scaler = label_scaler
        self.save_hyperparameters('params')

    def training_step(self, data, idx) -> Tensor:
        pred = self.model(data.x, data.edge_index, data.batch)
        scaled_preds = self.label_scaler.inverse_transform(pred).flatten()
        loss = self._report_loss(scaled_preds, data.y, 'train')
        return loss

    def validation_step(self, data, idx) -> None:
        pred = self.model(data.x, data.edge_index, data.batch)
        scaled_preds = self.label_scaler.inverse_transform(pred).flatten()
        self._report_loss(scaled_preds, data.y, 'val')
        self.val_metrics.update(scaled_preds, data.y)

    def validation_epoch_end(self, outputs) -> None:
        self.log_dict(detach_metrics(self.val_metrics.compute()))
        self.val_metrics.reset()

    def test_step(self, data, idx) -> None:
        encodings = self.model.encode(data.x, data.edge_index, data.batch)
        preds = self.model.readout(encodings)
        scaled_preds = self.label_scaler.inverse_transform(preds).flatten()
        self.test_metrics.update(scaled_preds, data.y)
        self.explainability_metrics.update(encodings, data.y)

    def test_epoch_end(self, outputs) -> None:
        test_metrics = self.test_metrics.compute()
        explainability_metrics = self.explainability_metrics.compute()
        self.test_results = detach_metrics(test_metrics | explainability_metrics)
        self.test_metrics.reset()
        self.explainability_metrics.reset()

    def configure_optimizers(self) -> dict[str, Any]:
        optimiser = Adam(self.model.parameters(), lr=self.params.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser,
            factor=DEFAULT_LR_PLATEAU_FACTOR,
            patience=DEFAULT_LR_PLATEAU_PATIENCE
        )
        return {
            'optimizer': optimiser,
            'lr_scheduler': scheduler,
            'monitor': 'loss_val',
        }

    def _report_loss(self, pred, y, prefix) -> Tensor:
        loss = self.loss(pred, y)
        self.log('loss_' + prefix, loss, batch_size=y.shape[0])
        return loss


def run_experiment(
    experiment_dir: Path,
    dataset: NamedLabelledDataset,
    architectures: List[GNNArchitecture],
    params: HyperParameters,
) -> None:
    """Perform a series of runs of different architectures and save the results"""
    torch.set_float32_matmul_precision(params.precision)
    proxies = {}
    metrics = {}
    for run_id, architecture in enumerate(architectures):
        AUTOML_LOGGER.debug(f"Running experiment on architecture {architecture}")
        run_proxies, run_metrics = perform_run(dataset, architecture, params, experiment_dir, run_name=str(run_id))
        proxies[str(architecture)] = analyse_results_distribution(run_proxies)
        metrics[str(architecture)] = analyse_results_distribution(run_metrics)
    save_experiment_results(proxies, experiment_dir, 'proxies')
    save_experiment_results(metrics, experiment_dir, 'metrics')


def perform_run(
    dataset: NamedLabelledDataset,
    architecture: GNNArchitecture,
    params: HyperParameters,
    experiment_dir: Optional[Path] = None,
    run_name: Optional[str] = None,
    calculate_proxies: bool = True,
) -> Tuple[dict[str, Metrics], dict[str, Metrics]]:
    """Perform multiple tests of an architecture across different data splits and random seeds."""
    save_logs = experiment_dir is not None
    if save_logs:
        run_dir = experiment_dir / (run_name if run_name is not None else generate_run_name())
    else:
        run_dir = None
    run_proxies = {}
    run_metrics = {}

    for base_seed in params.random_seeds:
        seed_generator = random.Random(base_seed)
        data_partitions = partition_dataset(dataset, params.dataset_split, base_seed)
        for data_version, (train_dataset, val_dataset, test_dataset) in data_partitions:
            seed = seed_generator.randint(MIN_SEED, MAX_SEED)
            tl.seed_everything(seed, workers=True)
            version = f'S{seed}_D{data_version}'
            model = GNN(architecture)
            datamodule = LightningDataset(
                train_dataset, val_dataset, test_dataset,
                batch_size=params.batch_size, num_workers=params.num_workers,
                pin_memory=not train_dataset.data.is_cuda
            )
            if calculate_proxies:
                run_proxies[version] = detach_metrics(DEFAULT_PROXIES(model, dataset))
            run_metrics[version] = train_model(
                model, params, datamodule, dataset.label_scaler, run_dir,
                version=version, save_logs=save_logs, save_checkpoints=save_logs,
            )
    if save_logs:
        save_run_results(run_metrics, run_dir, 'metrics')
        if calculate_proxies:
            save_run_results(run_proxies, run_dir, 'proxies')

    return (run_proxies, run_metrics) if calculate_proxies else run_metrics


def train_model(
    model: GNN,
    params: HyperParameters,
    datamodule: LightningDataModule,
    label_scaler: Scaler,
    run_dir: Optional[Path] = None,
    version: Optional[int | str] = None,
    save_logs: bool = True,
    save_checkpoints: bool = True,
    test_on_validation: bool = False,  # If test data is needed after further optimisation
) -> Metrics:
    """Train a model and return the testing metrics.

    Args:
        model: The model to train.
        params: The hyperparameters that define how the model is trained.
        datamodule: The data to train, validate and test the model on.
        label_scaler: The label scaler fit to the dataset.
        run_dir: The directory to save training logs to.
        version: The name of the fold in run_dir to save logs to.
        save_logs: Whether to save training logs.
        save_checkpoints: Whether to save the final model checkpoint.
        test_on_validation: Should testing dataloader or validation dataloader be used to report result.
    Returns:
        Metrics: The final performance of the model on the validation/testing dataset.
    """
    lit_model = LitGNN(model, params, DEFAULT_ACCURACY_METRICS, DEFAULT_EXPLAINABILITY_METRICS, label_scaler)

    callbacks = []
    if params.early_stop_patience > 0:
        callbacks.append(EarlyStopping(
            monitor='loss_val',
            mode='min',
            patience=params.early_stop_patience,
            min_delta=params.early_stop_min_delta
        ))
    if save_checkpoints:
        callbacks.append(ModelCheckpoint(
            # filename='{epoch:02d}-{loss_val:.2f}',
            # monitor='loss_val',
            save_last=True,
            save_top_k=0
        ))

    if save_logs:
        logger = TensorBoardLogger(
            save_dir=run_dir,
            name='',
            version=version
        )
    else:
        logger = False

    trainer = tl.Trainer(
        default_root_dir=run_dir,
        deterministic=True,
        accelerator='gpu',
        devices=1,
        log_every_n_steps=5,
        max_epochs=params.max_epochs,
        enable_checkpointing=save_checkpoints,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=False,
        enable_model_summary=save_logs,
    )

    trainer.fit(lit_model, datamodule=datamodule)
    test_dataloader = datamodule.val_dataloader() if test_on_validation else datamodule.test_dataloader()
    trainer.test(lit_model, test_dataloader)
    result = lit_model.test_results

    # Free-up memory
    del lit_model

    return result
