import pandas as pd

from HeLU.model.HeLU_model import HeLU_Crypto
from HeLU.callback import logger_callback
import lightning as pl
import torch
import yaml
from HeLU.dataset.crypto_dataset import ParquetDataset
from torch.utils.data import DataLoader

def configure(config_path = 'config/config.yaml'):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return config_dict

def data_split(Train):
    train_len = int(0.8 * len(Train))
    return Train[:train_len], Train[train_len:]

'''

def train():
    config_dict = configure()

    model_config = config_dict['model']
    train_config = config_dict['train']

    model = HeLU_Crypto(model_config)

    trainer = pl.Trainer(
        devices='mps',
        max_epochs=train_config['epochs'],
        progress_bar_refresh_rate=0,
        logger=True,
        enable_progress_bar=True
    )

    trainer.fit(model, DataLoader(dataset, batch_size= 32, shuffle=False),
                DataLoader(dataset, batch_size= 32, shuffle=False))
'''
# def evaluate():
#     config_dict = configure()
#
#     model_config = config_dict['model']
#     train_config = config_dict['train']
#
#     # model = make_model(*model_config)
#     crypto = HeLU_Crypto(*model_config)
#
#     trainer = pl.Trainer(train_config)
#
#     trainer.fit(model)

if __name__ == '__main__':
    dataset = ParquetDataset('/Users/wangyuhao/Desktop/drw-crypto-market-prediction/train.parquet')
    testing_dataset = pd.read_parquet('/Users/wangyuhao/Desktop/drw-crypto-market-prediction/test.parquet')
    config_dict = configure()

    model_config = config_dict['model']
    train_config = config_dict['train']

    model = HeLU_Crypto(model_config)

    trainer = pl.Trainer(
        accelerator='cpu',
        devices=1,
        max_epochs=train_config['epochs'],
        logger=True,
        enable_progress_bar=True,
    )

    trainer.fit(model, train_dataloaders = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0))
