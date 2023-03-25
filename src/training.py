from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Type

import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import MSELoss
from torch.optim import Adam
import pytorch_lightning as tl
from torch_geometric.data import LightningDataset
from torch_geometric.data.lightning_datamodule import LightningDataModule
from torchmetrics import MetricCollection

from src.config import LOG_DIR, DEFAULT_LOGGER, DEFAULT_LR_PLATEAU_PATIENCE, DEFAULT_LR_PLATEAU_FACTOR
from src.data.scaling import Scaler
from src.data.utils import DatasetSplit, NamedLabelledDataset, partition_dataset
from src.metrics import DEFAULT_METRICS, analyse_results_distribution
from src.models import GNNArchitecture, GNNModule, GNN
from src.reporting import generate_experiment_dir, generate_run_name, save_experiment_results, \
    save_run_results


@dataclass
class HyperParameters:
    random_seeds: List[int]
    dataset_split: DatasetSplit
    label_scaler: Type[Scaler]
    limit_batches: float
    batch_size: int
    early_stop_patience: int
    early_stop_min_delta: float
    lr: float
    max_epochs: int
    num_workers: int


class LitGNN(tl.LightningModule):
    def __init__(self, model: GNNModule, params: HyperParameters, metrics: MetricCollection, label_scaler: Scaler):
        super().__init__()
        self.model = model
        self.params = params
        self.loss = MSELoss()
        self.val_metrics = metrics.clone()
        self.test_metrics = metrics.clone()
        self.test_results = None
        self.label_scaler = label_scaler
        self.save_hyperparameters("params")

    def training_step(self, data, idx):
        pred = self.model(data.x, data.edge_index, data.batch)
        scaled_preds = self.label_scaler.inverse_transform(pred).flatten()
        loss = self._report_loss(scaled_preds, data.y, 'train')
        return loss

    def validation_step(self, data, idx):
        pred = self.model(data.x, data.edge_index, data.batch)
        scaled_preds = self.label_scaler.inverse_transform(pred).flatten()
        self._report_loss(scaled_preds, data.y, 'val')
        self.val_metrics.update(scaled_preds, data.y)

    def validation_epoch_end(self, outputs):
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_step(self, data, idx):
        pred = self.model(data.x, data.edge_index, data.batch)
        scaled_preds = self.label_scaler.inverse_transform(pred).flatten()
        self.test_metrics.update(scaled_preds, data.y)

    def test_epoch_end(self, outputs):
        self.test_results = self.test_metrics.compute()
        self.test_metrics.reset()

    def configure_optimizers(self):
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

    def _report_loss(self, pred, y, prefix):
        loss = self.loss(pred, y)
        self.log('loss_' + prefix, loss, batch_size=y.shape[0])
        return loss


def run_experiment(
        experiment_name: str,
        dataset: NamedLabelledDataset,
        label_scaler: Scaler,
        architectures: List[GNNArchitecture],
        params: HyperParameters,
        precision: str,
):
    """Perform a series of runs of different architectures and save the results"""
    torch.set_float32_matmul_precision(precision)
    experiment_dir = LOG_DIR / generate_experiment_dir(dataset, experiment_name)
    results = {}
    for architecture in architectures:
        DEFAULT_LOGGER.debug(f"Running experiment on architecture {architecture}")
        run_results = perform_run(dataset, label_scaler, architecture, params, experiment_dir)
        results[str(architecture)] = analyse_results_distribution(run_results)
    save_experiment_results(results, experiment_dir)


def perform_run(
    dataset: NamedLabelledDataset,
    label_scaler: Scaler,
    architecture: GNNArchitecture,
    params: HyperParameters,
    experiment_dir: Path,
    run_name: Optional[str] = None,
):
    """Perform multiple runs using k-fold cross validation and return the average results"""
    run_dir = experiment_dir / (run_name if run_name else generate_run_name())
    run_results = {}
    for seed in params.random_seeds:
        data_partitions = partition_dataset(dataset.dataset, params.dataset_split, seed)
        for data_version, (train_dataset, val_dataset, test_dataset) in data_partitions:
            tl.seed_everything(seed, workers=True)
            version = f'S{seed}_D{data_version}'
            model = GNN(architecture)
            datamodule = LightningDataset(
                train_dataset, val_dataset, test_dataset,
                batch_size=params.batch_size, num_workers=params.num_workers
            )
            result = train_model(model, params, datamodule, label_scaler, run_dir, version=version)
            run_results[version] = result

    save_run_results(run_results, run_dir)
    return run_results


def train_model(
    model: GNNModule,
    params: HyperParameters,
    datamodule: LightningDataModule,
    label_scaler: Scaler,
    run_dir: Path,
    version: Optional[int | str] = None,
    save_logs: bool = True,
    save_checkpoints: bool = True,
    test_on_validation: bool = False,  # If test data is needed after further optimisation
):
    model = LitGNN(model, params, DEFAULT_METRICS, label_scaler)

    callbacks = [EarlyStopping(
        monitor='loss_val',
        mode='min',
        patience=params.early_stop_patience,
        min_delta=params.early_stop_min_delta
    )]

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
        limit_train_batches=params.limit_batches,
        limit_test_batches=params.limit_batches,
        limit_val_batches=params.limit_batches,
    )

    trainer.fit(model, datamodule=datamodule)
    test_dataloader = datamodule.val_dataloader() if test_on_validation else datamodule.test_dataloader()
    trainer.test(model, test_dataloader)
    result = model.test_results

    # Free-up memory
    del model

    return result
