import abc
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import (
    EVAL_DATALOADERS,
    TRAIN_DATALOADERS,
)
from torch import nn
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from src.dataset.inference_dataset import InferenceDataset
from src.extractor.base_extractor import ExtractorNetwork
from src.metrics.metrics import SHRECMetricEvaluator
from src.dataset.text_pc_dataset import TextPointCloudDataset
from src.utils.pc_transform import (
    Normalize,
    RandRotation_z,
    RandomNoise,
    ToTensor,
)


class AbstractModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None
        self.metric_evaluator = None
        self.init_model()

    def setup(self, stage):
        if stage in ["fit", "validate", "test"]:
            train_transforms = transforms.Compose(
                [Normalize(), RandRotation_z(), RandomNoise(), ToTensor()]
            )

            validation_transforms = transforms.Compose([Normalize(), ToTensor()])

            self.train_dataset = TextPointCloudDataset(
                pc_transform=train_transforms,
                **self.cfg["dataset"]["train"]["params"],
            )

            self.val_dataset = TextPointCloudDataset(
                pc_transform=validation_transforms,
                **self.cfg["dataset"]["val"]["params"],
            )

            self.metric_evaluator = SHRECMetricEvaluator(
                embed_dim=self.cfg["model"]["embed_dim"]
            )
        elif stage == "predict":
            test_transforms = transforms.Compose([Normalize(), ToTensor()])

            test_set = InferenceDataset(
                root_dir="../dataset", pc_transform=test_transforms, stage="train"
            )
            test_loader = DataLoader(dataset=test_set, batch_size=1)

    @abc.abstractmethod
    def init_model(self):
        """
        Function to initialize model
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, batch):
        raise NotImplementedError

    @abc.abstractmethod
    def compute_loss(self, forwarded_batch, input_batch):
        """
        Function to compute loss
        Args:
            forwarded_batch: output of `forward` method
            input_batch: input of batch method

        Returns:
            loss: computed loss
        """
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        # 1. get embeddings from model
        forwarded_batch = self.forward(batch)
        # 2. Calculate loss
        loss = self.compute_loss(forwarded_batch=forwarded_batch, input_batch=batch)
        # 3. Update monitor
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # 1. Get embeddings from model
        forwarded_batch = self.forward(batch)
        # 2. Calculate loss
        loss = self.compute_loss(forwarded_batch=forwarded_batch, input_batch=batch)
        # 3. Update metric for each batch
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.metric_evaluator.append(
            g_emb=forwarded_batch["pc_embedding_feats"].float(),
            q_emb=forwarded_batch["query_embedding_feats"].float(),
            query_ids=batch["query_ids"],
            gallery_ids=batch["point_cloud_ids"],
            target_ids=batch["point_cloud_ids"],
        )

        return {"loss": loss}

    def validation_epoch_end(self, outputs) -> None:
        """
        Callback at validation epoch end to do additional works
        with output of validation step, note that this is called
        before `training_epoch_end()`
        Args:
            outputs: output of validation step
        """
        self.log_dict(
            self.metric_evaluator.evaluate(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.metric_evaluator.reset()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_loader = DataLoader(
            dataset=self.train_dataset,
            collate_fn=self.train_dataset.collate_fn,
            **self.cfg["data_loader"]["train"]["params"],
        )
        return train_loader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        val_loader = DataLoader(
            dataset=self.val_dataset,
            collate_fn=self.val_dataset.collate_fn,
            **self.cfg["data_loader"]["val"]["params"],
        )
        return val_loader

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), self.cfg["trainer"]["lr"], weight_decay=0.0001
        )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[120, 250, 300], gamma=0.5
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, extractor: ExtractorNetwork, latent_dim=128):
        super().__init__()
        self.extractor = extractor
        self.feature_dim = extractor.feature_dim
        # mlp = 4 layers, feature_dim -> feature_dim / 2 -> feature_dim / 4 -> latent_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 2, self.feature_dim // 4),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 4, latent_dim),
        )

    def forward(self, x):
        x = self.extractor.get_embedding(x)
        x = self.mlp(x)
        return x
