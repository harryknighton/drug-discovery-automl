import logging
from typing import List

import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.nn import MSELoss
from torch.optim import AdamW
import pytorch_lightning as tl
from torch_geometric.data import LightningDataset
from torch_geometric.data.lightning_datamodule import LightningDataModule
from torchmetrics import MetricCollection

from src.config import DATA_DIR, NUM_WORKERS, DEFAULT_METRICS, LOG_DIR
from src.data import split_dataset, partition_dataset, HTSDataset
from src.models import construct_gnn, construct_mlp, HyperParameters, ModelArchitecture, GNNArchitecture
from src.reporting import generate_experiment_dir, generate_run_name, save_run, save_experiment_results


class LitGNN(tl.LightningModule):
    def __init__(self, architecture: GNNArchitecture, params: HyperParameters, metrics: MetricCollection):
        super().__init__()
        self.gnn = construct_gnn(architecture)
        self.regression_mlp = construct_mlp(architecture.regression_layer)
        self.params = params
        self.loss = MSELoss()
        self.val_metrics = metrics.clone()
        self.test_metrics = metrics.clone()
        self.test_results = None

    def forward(self, x, edge_index, batch):
        embedding = self.gnn(x, edge_index, batch)
        return self.regression_mlp(embedding)

    def training_step(self, data, idx):
        pred = self.forward(data.x, data.edge_index, data.batch)
        loss = self._report_loss(pred.flatten(), data.y, 'train')
        return loss

    def validation_step(self, data, idx):
        pred = self.forward(data.x, data.edge_index, data.batch).flatten()
        self._report_loss(pred, data.y, 'val')
        self.val_metrics.update(pred, data.y)

    def validation_epoch_end(self, outputs):
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_step(self, data, idx):
        pred = self.forward(data.x, data.edge_index, data.batch).flatten()
        self.test_metrics.update(pred, data.y)

    def test_epoch_end(self, outputs):
        self.test_results = self.test_metrics.compute()
        self.test_metrics.reset()

    def configure_optimizers(self):
        optimiser = AdamW(self.gnn.parameters(), lr=self.params.lr)
        return optimiser

    def _report_loss(self, pred, y, prefix):
        # TODO: Remove slash from name
        loss = self.loss(pred, y)
        self.log('loss/' + prefix, loss, batch_size=y.shape[0])
        return loss


def run_experiment(experiment_name: str, dataset_name: str, architectures: List[GNNArchitecture], params: HyperParameters, random_seeds: List[int]):
    """Perform a series of runs of different architectures and save the results"""
    experiment_dir = LOG_DIR / generate_experiment_dir(dataset_name, params.use_sd_readouts, experiment_name)
    dataset_dir = DATA_DIR / dataset_name
    dataset = HTSDataset(dataset_dir, 'DR')  # TODO: Add SD/DR dataset

    logging.info("Running experiment at " + str(experiment_dir))
    results = {}
    for architecture in architectures:
        architecture_results = {}
        for seed in random_seeds:
            tl.seed_everything(seed, workers=True)
            params.random_seed = seed
            run_result = perform_run(dataset, architecture, params, experiment_dir)
            architecture_results[seed] = run_result
        results[str(architecture)] = architecture_results
    save_experiment_results(results, experiment_dir)


def perform_run(dataset: HTSDataset, architecture: GNNArchitecture, params: HyperParameters, experiment_dir):
    """Perform multiple runs using k-fold cross validation and return the average results"""
    run_dir = experiment_dir / generate_run_name()
    trial_results = []
    for train_dataset, val_dataset, test_dataset in partition_dataset(dataset, params):
        datamodule = LightningDataset(
            train_dataset, val_dataset, test_dataset,
            batch_size=params.batch_size, num_workers=NUM_WORKERS
        )
        result = train_model(architecture, params, datamodule, run_dir)
        trial_results.append(result)

    result = _calculate_run_result(trial_results)
    save_run(result, architecture, params, run_dir)
    return result


def train_model(architecture: GNNArchitecture, params: HyperParameters, datamodule: LightningDataModule, run_dir):
    model = LitGNN(architecture, params, DEFAULT_METRICS)

    checkpoint_callback = ModelCheckpoint(
        monitor='loss/val',
        mode='min',
        filename='{epoch:02d}-{loss/val:.2f}',
        save_top_k=2,
    )

    early_stop_callback = EarlyStopping(
        monitor='loss/val',
        mode='min',
        patience=params.early_stop_patience,
        min_delta=params.early_stop_min_delta
    )

    trainer = tl.Trainer(
        default_root_dir=run_dir,
        deterministic=True,
        accelerator='gpu',
        devices=1,
        log_every_n_steps=1,
        max_epochs=params.max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    trainer.fit(model, datamodule=datamodule)

    trainer.test(ckpt_path='best', datamodule=datamodule)
    result = model.test_results
    model.test_results = None  # Free-up memory

    return result


def _calculate_run_result(trial_results):
    """Calculate the mean and variance of the results of all trials"""
    assert trial_results is not None
    stacked_metrics = {name: np.array([float(result[name]) for result in trial_results]) for name in trial_results[0]}
    means = {name: float(np.mean(metrics)) for name, metrics in stacked_metrics.items()}
    variances = {name: float(np.var(metrics)) for name, metrics in stacked_metrics.items()}
    metrics = {name: {'mean': means[name], 'variance': variances[name]} for name in stacked_metrics}
    return metrics
