import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import pandas as pd
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
        self.log('train_loss', loss, prog_bar = True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        mask = make_std_mask(x, self.batch_size)
        y_hat = self(x, mask, mask)
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        predictions = self(x)
        return predictions

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

