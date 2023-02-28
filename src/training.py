from pathlib import Path
from typing import List, Optional

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
from src.data import partition_dataset, HTSDataset
from src.metrics import DEFAULT_METRICS, StandardScaler, analyse_results_distribution
from src.models import construct_gnn, construct_mlp, GNNArchitecture
from src.parameters import HyperParameters, DatasetUsage
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
        self.save_hyperparameters("architecture", "params")

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
        dataset_name: str,
        architectures: List[GNNArchitecture],
        params: HyperParameters,
        random_seeds: List[int],
        precision: str,
        sd_ckpt_path: Optional[Path] = None
):
    """Perform a series of runs of different architectures and save the results"""
    torch.set_float32_matmul_precision(precision)
    experiment_dir = LOG_DIR / generate_experiment_dir(dataset_name, params.dataset_usage, experiment_name)
    DEFAULT_LOGGER.info(f"Running experiment {experiment_name} at {experiment_dir}")
    DEFAULT_LOGGER.info(f"Loading dataset {dataset_name} containing {params.dataset_usage.name}")
    dataset = HTSDataset(dataset_name, params.dataset_usage)
    if params.dataset_usage == DatasetUsage.DRWithSDReadouts:
        sd_model = LitGNN.load_from_checkpoint(sd_ckpt_path, label_scaler=dataset.scaler, metrics=DEFAULT_METRICS)
        dataset.augment_dataset_with_sd_readouts(sd_model)
    results = {}
    for architecture in architectures:
        DEFAULT_LOGGER.debug(f"Running experiment on architecture {architecture}")
        trials_results = []
        for seed in random_seeds:
            tl.seed_everything(seed, workers=True)
            params.random_seed = seed
            run_result = perform_run(dataset, architecture, params, experiment_dir)
            trials_results.extend(list(run_result.values()))
        architecture_results = analyse_results_distribution(trials_results)
        results[str(architecture)] = architecture_results
    save_experiment_results(results, experiment_dir)


def perform_run(
    dataset: HTSDataset,
    architecture: GNNArchitecture,
    params: HyperParameters,
    experiment_dir: Path,
    run_name: Optional[str] = None,
):
    """Perform multiple runs using k-fold cross validation and return the average results"""
    run_dir = experiment_dir / (run_name if run_name else generate_run_name())
    trial_results = {}
    for version, (train_dataset, val_dataset, test_dataset) in partition_dataset(dataset, params):
        datamodule = LightningDataset(
            train_dataset, val_dataset, test_dataset,
            batch_size=params.batch_size, num_workers=params.num_workers
        )
        result = train_model(architecture, params, datamodule, dataset.scaler, run_dir, version=version)
        trial_results[version] = result

    save_run(trial_results, architecture, params, run_dir)
    return trial_results


def train_model(
    architecture: GNNArchitecture,
    params: HyperParameters,
    datamodule: LightningDataModule,
    label_scaler: StandardScaler,
    run_dir: Path,
    version: Optional[int] = None,
    save_logs: bool = True,
    save_checkpoints: bool = True,
    test_on_validation: bool = False,  # If test data is needed after further optimisation
):
    model = LitGNN(architecture, params, DEFAULT_METRICS, label_scaler)

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
