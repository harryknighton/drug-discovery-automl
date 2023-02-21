import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import MSELoss
from torch.optim import Adam
import pytorch_lightning as tl
from torch_geometric.data import LightningDataset
from torch_geometric.data.lightning_datamodule import LightningDataModule
from torchmetrics import MetricCollection

from src.config import LOG_DIR
from src.data import partition_dataset, HTSDataset, DatasetUsage
from src.metrics import DEFAULT_METRICS, StandardScaler
from src.models import construct_gnn, construct_mlp, GNNArchitecture
from src.parameters import HyperParameters
from src.reporting import generate_experiment_dir, generate_run_name, save_run, save_experiment_results


class LitGNN(tl.LightningModule):
    def __init__(self, architecture: GNNArchitecture, params: HyperParameters, metrics: MetricCollection, label_scaler: StandardScaler):
        super().__init__()
        self.gnn = construct_gnn(architecture)
        self.regression_mlp = construct_mlp(architecture.regression_layer)
        self.params = params
        self.loss = MSELoss()
        self.val_metrics = metrics.clone()
        self.test_metrics = metrics.clone()
        self.test_results = None
        self.label_scaler = label_scaler

    def forward(self, x, edge_index, batch):
        embedding = self.gnn(x, edge_index, batch)
        return self.regression_mlp(embedding)

    def training_step(self, data, idx):
        pred = self.forward(data.x, data.edge_index, data.batch)
        loss = self._report_loss(pred.flatten(), data.y[:, 0], 'train')  # Calculate loss on scaled labels
        return loss

    def validation_step(self, data, idx):
        pred = self.forward(data.x, data.edge_index, data.batch)
        self._report_loss(pred.flatten(), data.y[:, 0], 'val')  # Calculate loss on scaled labels
        self.val_metrics.update(self.label_scaler.inverse_transform(pred).flatten(), data.y[:, 1])

    def validation_epoch_end(self, outputs):
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_step(self, data, idx):
        pred = self.forward(data.x, data.edge_index, data.batch)
        self.test_metrics.update(self.label_scaler.inverse_transform(pred).flatten(), data.y[:, 1])

    def test_epoch_end(self, outputs):
        self.test_results = self.test_metrics.compute()
        self.test_metrics.reset()

    def configure_optimizers(self):
        optimiser = Adam(self.parameters(), lr=self.params.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.5, patience=20)
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
        dataset_name: str,
        dataset_usage: DatasetUsage,
        architectures: List[GNNArchitecture],
        params: HyperParameters,
        random_seeds: List[int],
        precision: str
):
    """Perform a series of runs of different architectures and save the results"""
    torch.set_float32_matmul_precision(precision)
    experiment_dir = LOG_DIR / generate_experiment_dir(dataset_name, dataset_usage, experiment_name)
    dataset = HTSDataset(dataset_name, dataset_usage)
    logging.info(f"Running experiment {experiment_name} at {experiment_dir}")
    results = {}
    for architecture in architectures:
        logging.info(f"Running on architecture {architecture}")
        architecture_results = {}
        for seed in random_seeds:
            tl.seed_everything(seed, workers=True)
            params.random_seed = seed
            run_result = perform_run(dataset, architecture, params, experiment_dir)
            architecture_results[seed] = run_result
        results[str(architecture)] = architecture_results
    save_experiment_results(results, experiment_dir)


def perform_run(
    dataset: HTSDataset,
    architecture: GNNArchitecture,
    params: HyperParameters,
    experiment_dir: Path
):
    """Perform multiple runs using k-fold cross validation and return the average results"""
    run_dir = experiment_dir / generate_run_name()
    trial_results = {}
    for version, (train_dataset, val_dataset, test_dataset) in partition_dataset(dataset, params):
        datamodule = LightningDataset(
            train_dataset, val_dataset, test_dataset,
            batch_size=params.batch_size, num_workers=params.num_workers
        )
        result = train_model(architecture, params, datamodule, dataset.scaler, run_dir, version=version)
        trial_results[version] = result

    result = _calculate_run_result(trial_results)
    save_run(trial_results, architecture, params, run_dir)
    return result


def train_model(
    architecture: GNNArchitecture,
    params: HyperParameters,
    datamodule: LightningDataModule,
    label_scaler: StandardScaler,
    run_dir: Path,
    version: Optional[int] = None
):
    model = LitGNN(architecture, params, DEFAULT_METRICS, label_scaler)

    checkpoint_callback = ModelCheckpoint(
        filename='{epoch:02d}-{loss_val:.2f}',
        monitor='loss_val',
        mode='min',
        save_top_k=1,
    )

    early_stop_callback = EarlyStopping(
        monitor='loss_val',
        mode='min',
        patience=params.early_stop_patience,
        min_delta=params.early_stop_min_delta
    )

    logger = TensorBoardLogger(
        save_dir=run_dir,
        name='',
        version=version
    )

    trainer = tl.Trainer(
        default_root_dir=run_dir,
        deterministic=True,
        accelerator='gpu',
        devices=1,
        log_every_n_steps=1,
        max_epochs=params.max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        enable_progress_bar=False,
        enable_model_summary=True,
    )

    trainer.fit(model, datamodule=datamodule)

    trainer.test(ckpt_path='best', datamodule=datamodule)
    result = model.test_results
    model.test_results = None  # Free-up memory

    return result


def _calculate_run_result(trial_results: dict[int, dict]) -> dict[str, dict]:
    """Calculate the mean and variance of the results of all trials"""
    assert trial_results is not None
    results = list(trial_results.values())
    stacked_metrics = {name: np.array([float(result[name]) for result in results]) for name in results[0]}
    means = {name: float(np.mean(metrics)) for name, metrics in stacked_metrics.items()}
    variances = {
        name: float(np.var(metrics, ddof=1)) if len(metrics) > 1 else 0.
        for name, metrics in stacked_metrics.items()
    }
    metrics = {name: {'mean': means[name], 'variance': variances[name]} for name in stacked_metrics}
    return metrics
