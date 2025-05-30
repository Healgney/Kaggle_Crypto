import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import pandas as pd

from glob import glob

class Crypto_dataset(Dataset):

    """
    自定义Dataset。每次返回一个(x, y)对。
    输入:
            data_x: numpy array 或 torch.Tensor，shape=(样本数, 特征数)
            data_y: numpy array 或 torch.Tensor，shape=(样本数, 1)
    """
    def __init__(self, data_path = '/kaggle_crypto_data'):
        '''
        :param data_x: [N, feature_dim]
        :param data_y: [N, 1]
        '''
        super().__init__()

        mat_file = glob(data_path + '/*.parquet')

        data = {}

        for f in mat_file:
            df = pd.read_parquet(f)
            name = f.split('/')[-1].split('.')[0]
            data[name] = df



        # self.x = torch.tensor(data_x, dtype = torch.float32)
        # self.y = torch.tensor(data_y, dtype = torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        '''
        :param idx: data's columns
        '''
        return self.x[idx], self.y[idx]


class Crypto(pl.LightningModule):
    def __init__(self, model, lr = 1e-3):
        '''
        :param model:
        :param lr: learning rate
        '''
        super().__init__()
        self.model = model
        self.lr = lr
        self.save_hyperparameters()  # 自动保存超参数，便于实验追踪和复现

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