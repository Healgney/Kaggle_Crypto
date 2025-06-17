import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.optim as optim
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.cosine_lr import CosineLRScheduler
import pandas as pd
import numpy as np
from glob import glob
from HeLU.model.model_factory import make_model, make_std_mask

def model_parameter(config):
    return make_model(**config)


class HeLU_Crypto(pl.LightningModule):
    def __init__(self, model_config:dict, lr = 1e-3):
        '''
        :param model_config:
        :param lr: learning rate
        '''
        super().__init__()
        self.model = model_parameter(model_config)
        self.lr = lr
        self.save_hyperparameters()
        self.time_window = model_config['local_context_length']
        self.batch_size = model_config['batch_size']

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # print(f'x: {x.shape}')
        mask = make_std_mask(x, self.batch_size)
        # print(f'mask: {mask.shape}')
        y_prediction = self(x,mask,mask)

        loss = F.mse_loss(y_prediction,y)
        self.log('train_loss', loss, prog_bar = True, logger=True)
        self.log_dict(
            {'lr': self.lr_schedulers().optimizer.param_groups[0]['lr']},
            logger=True, sync_dist=True, on_step=True, prog_bar=True, on_epoch=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        mask = make_std_mask(x, self.batch_size)
        y_hat = self(x, mask, mask)
        loss = F.mse_loss(y_hat, y)
        
        y_pred_demean = y_hat - torch.mean(y_hat)
        y_demean = y - torch.mean(y)
        corr = ((torch.sum(y_pred_demean * y_demean)) / torch.sqrt((torch.sum(y_pred_demean ** 2) * torch.sum(y_demean ** 2))))
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("IC", corr, prog_bar=True)
        return {"val_loss": loss, "IC": corr}

    def predict_step(self, batch, batch_idx):
        x = batch
        mask = make_std_mask(x, self.batch_size)
        predictions = self(x, mask, mask)
        return predictions

    def set_optimizer_config(
            self, learning_rate: float,
            weight_decay: float,
            learning_rate_min: float = 1e-5,
            t_initial: int = 50000,
            warmup_t: int = 2000
    ):
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._learning_rate_min = learning_rate_min
        self._t_initial = t_initial
        self._warmup_t = warmup_t

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self._weight_decay, eps=1e-6
        )

        scheduler = CosineLRScheduler(
            optimizer, t_initial=self._t_initial, lr_min=self._learning_rate_min, warmup_t=self._warmup_t,
            warmup_lr_init=1e-6,
        )
        super().configure_optimizers()

        # return optimizer
        return dict(
            optimizer=optimizer,
            lr_scheduler={
                "scheduler": scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1
            }
        )

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.global_step)

