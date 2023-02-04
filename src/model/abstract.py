import abc
from ctypes import Union
from typing import List, Optional
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS
import torch
from src.utils.device import detach

class AbstractModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
    
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
    
    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
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
        """
        Implement PyTorch DataLoaders for training.
        Returns:
            TRAIN_DATALOADERS: train dataloader
        """
        # TODO: add train dataloader
        pass
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        """
        Implement PyTorch DataLoaders for evaluation.
        Returns:
            EVAL_DATALOADERS: evaluation dataloader
        """
        pass
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), self.cfg.trainer["lr"])

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
          optimizer, 
          milestones=[120, 250, 300], 
        gamma=0.5)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
