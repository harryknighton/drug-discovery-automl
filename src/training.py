import logging
from typing import List

import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from torch.nn import MSELoss
from torch.optim import AdamW
import pytorch_lightning as tl
from torch_geometric.data import LightningDataset
from torch_geometric.data.lightning_datamodule import LightningDataModule

from src.config import DATA_DIR, NUM_WORKERS, DEFAULT_METRICS, LOG_DIR
from src.data import split_dataset, k_folds, HTSDataset
from src.models import construct_model, HyperParameters, ModelArchitecture
from src.reporting import generate_experiment_dir, generate_run_name, save_run, save_experiment_results


class LitGNN(tl.LightningModule):
    def __init__(self, architecture, params, metrics):
        super().__init__()
        self.gnn = construct_model(architecture)
        self.params = params
        self.loss = MSELoss()
        self.val_metrics = metrics.clone()
        self.test_metrics = metrics.clone()
        self.test_results = None

    def training_step(self, data, idx):
        pred = self.gnn(data.x, data.edge_index, data.batch)
        pred = pred.flatten()
        loss = self._report_loss(pred, data.y, 'train')
        return loss

    def validation_step(self, data, idx):
        pred = self.gnn(data.x, data.edge_index, data.batch)
        pred = pred.flatten()
        self._report_loss(pred, data.y, 'val')
        self.val_metrics.update(pred, data.y)

    def validation_epoch_end(self, outputs):
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_step(self, data, idx):
        pred = self.gnn(data.x, data.edge_index, data.batch)
        pred = pred.flatten()
        self.test_metrics.update(pred, data.y)

    def test_epoch_end(self, outputs):
        self.test_results = self.test_metrics.compute()
        self.test_metrics.reset()

    def configure_optimizers(self):
        optimiser = AdamW(self.gnn.parameters(), lr=self.params.lr)
        return optimiser

    def _report_loss(self, pred, y, prefix):
        loss = self.loss(pred, y)
        self.log('loss/' + prefix, loss, batch_size=y.shape[0])
        return loss


def run_experiment(experiment_name: str, dataset_name: str, architectures: List[ModelArchitecture], params: HyperParameters, random_seeds: List[int]):
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
            if params.k_folds == 1:
                run_result = perform_run(dataset, architecture, params, experiment_dir)
            elif params.k_folds > 1:
                run_result = perform_k_fold_run(dataset, architecture, params, experiment_dir)
            else:
                raise ValueError(f"k_folds={params.k_folds} must be positive")
            architecture_results[seed] = run_result
        results[str(architecture)] = architecture_results
    save_experiment_results(results, experiment_dir)


def perform_run(dataset: HTSDataset, architecture: ModelArchitecture, params: HyperParameters, experiment_dir):
    """Perform a single run over the dataset"""
    run_dir = experiment_dir / generate_run_name()
    test_dataset, training_dataset = split_dataset(dataset, params.test_split)
    train_dataset, val_dataset = split_dataset(dataset, params.train_val_split)
    datamodule = LightningDataset(
        train_dataset, val_dataset, test_dataset,
        batch_size=params.batch_size, num_workers=NUM_WORKERS
    )
    result = train_model(architecture, params, datamodule, run_dir)
    result = {key: {'mean': float(value)} for key, value in result.items()}  # Conform to write_experiment_results()
    save_run(result, architecture, params, run_dir)
    return result


def perform_k_fold_run(dataset: HTSDataset, architecture: ModelArchitecture, params: HyperParameters, experiment_dir):
    """Perform multiple runs using k-fold cross validation and return the average results"""
    run_dir = experiment_dir / generate_run_name()
    training_dataset, test_dataset = split_dataset(dataset, params.test_split)

    fold_results = []
    for train_fold, val_fold in k_folds(training_dataset, params.k_folds, params.random_seed):
        datamodule = LightningDataset(train_fold, val_fold, test_dataset, batch_size=params.batch_size, num_workers=NUM_WORKERS)
        result = train_model(architecture, params, datamodule, run_dir)
        fold_results.append(result)

    result = _calculate_k_fold_result(fold_results)
    save_run(result, architecture, params, run_dir)
    return result


def train_model(architecture: ModelArchitecture, params: HyperParameters, datamodule: LightningDataModule, run_dir):
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
        enable_model_summary=True,
    )

    trainer.fit(model, datamodule=datamodule)

    trainer.test(ckpt_path='best', datamodule=datamodule)
    result = model.test_results
    model.test_results = None  # Free-up memory

    return result


def _calculate_k_fold_result(fold_results):
    """Calculate the mean and variance of the results of the k runs"""
    stacked_metrics = {name: np.array([float(fold[name]) for fold in fold_results]) for name in fold_results[0]}
    means = {name: float(np.mean(metrics)) for name, metrics in stacked_metrics.items()}
    variances = {name: float(np.var(metrics)) for name, metrics in stacked_metrics.items()}
    metrics = {name: {'mean': means[name], 'variance': variances[name]} for name in stacked_metrics}
    return metrics
