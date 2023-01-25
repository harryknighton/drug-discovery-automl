from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.nn import MSELoss
from torch.optim import AdamW
import pytorch_lightning as tl
from torch_geometric.loader import DataLoader

from src.config import LOG_DIR, DATA_DIR, NUM_WORKERS, DEFAULT_METRICS
from src.data import DRDataset, split_dataset, k_folds
from src.models import construct_model, HyperParameters
from src.reporting import generate_experiment_path, generate_run_name, save_k_fold_metrics, save_architecture, \
    save_hyper_parameters


class LitGNN(tl.LightningModule):
    def __init__(self, architecture, metrics):
        super().__init__()
        self.gnn = construct_model(architecture)
        self.loss = MSELoss()
        self.val_metrics = metrics.clone()
        self.test_metrics = metrics.clone()
        self.test_results = None

    def training_step(self, data, idx):
        pred = self.gnn(data.x, data.edge_index, data.batch)
        self._report_loss(pred, data.y, 'train')

    def validation_step(self, data, idx):
        pred = self.gnn(data.x, data.edge_index, data.batch)
        self._report_loss(pred, data.y, 'val')
        self.val_metrics.update(pred, data.y)

    def validation_step_end(self, outputs):
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_step(self, data, idx):
        pred = self.gnn(data.x, data.edge_index, data.batch)
        self.test_metrics.update(pred, data.y)

    def test_epoch_end(self, outputs):
        self.test_results = self.test_metrics.compute()
        self.test_metrics.reset()

    def configure_optimizers(self):
        optimiser = AdamW(self.gnn.parameters())
        return optimiser

    def _report_loss(self, pred, y, prefix):
        loss = self.loss(pred, y)
        self.log('loss/' + prefix, loss, logger=True, on_step=True, batch_size=y.shape[0])


def run_experiment(architecture, params: HyperParameters, dataset_name: str):
    tl.seed_everything(params.random_seed, workers=True)

    dataset_dir = DATA_DIR / DATASET_NAME
    experiment_dir = LOG_DIR / generate_experiment_path(dataset_name, False)
    run_dir = experiment_dir / generate_run_name()

    dataset = DRDataset(root=dataset_dir)
    dataset.shuffle()
    training_dataset, test_dataset = split_dataset(dataset, params.train_test_split)
    test_dataloader = DataLoader(test_dataset, batch_size=params.batch_size, num_workers=NUM_WORKERS)
    fold_metrics = []
    for train_fold, val_fold in k_folds(training_dataset, params.k_folds, params.random_seed):
        training_dataloader = DataLoader(train_fold, batch_size=params.batch_size, shuffle=True, num_workers=NUM_WORKERS)
        validation_dataloader = DataLoader(val_fold, batch_size=params.batch_size, num_workers=NUM_WORKERS)

        model = LitGNN(architecture, DEFAULT_METRICS)

        checkpoint_callback = ModelCheckpoint(
            monitor='r',
            mode='max',
            filename='{epoch:02d}-{loss/val:.2f}',
            save_top_k=2,
        )

        early_stop_callback = EarlyStopping(
            monitor='r',
            mode='max',
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
        )

        trainer.fit(
            model,
            training_dataloader,
            validation_dataloader,
        )

        trainer.test(ckpt_path='best', dataloaders=test_dataloader)
        fold_metrics.append(trainer.callback_metrics)

    save_k_fold_metrics(fold_metrics, run_dir)
    save_architecture(architecture, run_dir)
    save_hyper_parameters(params, architecture, run_dir)
