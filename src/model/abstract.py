import abc
from ctypes import Union
from typing import List
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import (
    EPOCH_OUTPUT,
    EVAL_DATALOADERS,
    TRAIN_DATALOADERS,
)
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from src.dataset.base_dataset import SiameseNetworkDataset
from src.transformer.pc_transformer import Normalize, ToTensor
from src.utils.device import detach


class AbstractModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.init_model()

    def setup(self, stage):
        if stage != "predict":
            pc_transforms = transforms.Compose([Normalize()(), ToTensor()])

            img_transforms = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

            self.train_dataset = SiameseNetworkDataset(
                root_dir=self.cfg["dataset"],
                pc_transforms=pc_transforms,
                img_transforms=img_transforms,
            )

            self.val_dataset = SiameseNetworkDataset(
                root_dir=self.cfg["dataset"],
                pc_transforms=pc_transforms,
                img_transforms=img_transforms,
            )

    @abc.abstractmethod
    def init_model(self):
        """
        Function to initialize model
        """
        pass

    @abc.abstractmethod
    def forward(self, batch):
        raise NotImplementedError

    @abc.abstractmethod
    def compute_loss(self, batch, **kwargs):
        """
        Function to compute
        Args:
            batch (_type_): _description_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        # 1. get embeddings from model
        output = self.forward(batch)
        # 2. Calculate loss
        loss = self.compute_loss(**output, batch=batch)
        # 3. TODO: Update monitor

        return {"loss": detach(loss)}

    def validation_step(self, batch, batch_idx):
        # 1. Get embeddings from model
        output = self.forward(batch)
        # 2. Calculate loss
        loss = self.compute_loss(**output, batch=batch)
        # 3. TODO: Update metric for each batch

        return {"loss": detach(loss)}

    def validation_epoch_end(
        self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]
    ) -> None:
        """
        Callback at validation epoch end to do additional works
        with output of validation step, note that this is called
        before `training_epoch_end()`
        Args:
            outputs: output of validation step
        """
        # TODO: add metric evaluation and reset
        pass

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=self.train_dataset.collate_fn,
        )
        return train_loader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=self.val_dataset.collate_fn,
        )
        return val_loader

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), self.cfg.trainer["lr"])

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[120, 250, 300], gamma=0.5
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
