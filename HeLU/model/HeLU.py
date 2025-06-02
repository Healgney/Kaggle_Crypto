import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import pandas as pd
from glob import glob

from .model_factory import make_model

def model_factory(config):
    return make_model(**config)

class HeLU_Crypto(pl.LightningModule):
    def __init__(self, model_config:dict, lr = 1e-3):
        '''
        :param model:
        :param lr: learning rate
        '''
        super().__init__()
        self.model = model_factory(model_config)
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_prediction = self(x)
        loss = F.mse_loss(y_prediction,y)
        self.log('train_loss', loss, prog_bar = True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch
        y_hat = self(x)
        return y_hat

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

