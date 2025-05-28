import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F


class Mydataset(Dataset):
    """
    自定义Dataset。每次返回一个(x, y)
    输入:
            data_x: numpy array 或 torch.Tensor，shape=(样本数, 特征数)
            data_y: numpy array 或 torch.Tensor，shape=(样本数, 1)
    """
    def __init__(self, data_x, data_y):
        '''
        :param data_x: [N, feature_dim]
        :param data_y: [N, 1]
        '''
        super().__init__()
        self.x = torch.tensor(data_x, dtype = torch.float32)
        self.y = torch.tensor(data_y, dtype = torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        '''
        :param idx: data's columns
        '''
        return self.x[idx], self.y[idx]

'''
class MyDataModule(pl.LightningDataModule):
    def __init__(self, data_x, data_y, batch_size):
        
        super().__init__()
        self.data_x = data_x  # [N, feature_dim]
        self.data_y = data_y  # [N, 1] 或 [N]
        self.batch_size = batch_size
'''

class MyModel(pl.LightningModule):
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





