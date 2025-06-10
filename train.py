import pandas as pd

from HeLU.model.HeLU_model import HeLU_Crypto
from HeLU.callback import logger_callback
import lightning as pl
import torch
import yaml
from HeLU.dataset.crypto_dataset import ParquetDataset
from HeLU.logger.logger_factory import make_logger
from torch.utils.data import DataLoader

def configure(config_path = 'config/config.yaml'):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return config_dict


def data_split(Train):
    train_len = int(0.8 * len(Train))
    return Train[:train_len], Train[train_len:]


def train():
    config_dict = configure()
    model_config = config_dict['model']
    train_config = config_dict['train']
    dataset = ParquetDataset('/home/healgney/Downloads/train.parquet', model_config['window_size'])

    model = HeLU_Crypto(model_config)

    trainer = pl.Trainer(
        accelerator='cuda',
        # devices='cuda:0',
        max_epochs=train_config['epochs'],
        logger=make_logger(config_dict),
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    trainer.fit(model, train_dataloaders = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, drop_last=True))

def evaluate():
    config_dict = configure()
    model_config = config_dict['model']
    train_config = config_dict['train']
    dataset = ParquetDataset('/home/healgney/Downloads/test.parquet', model_config['window_size'])

    model = HeLU_Crypto(model_config)
    model = model.load_from_checkpoint("model.ckpt")

    trainer = pl.Trainer(
        accelerator='cuda',
        # devices='cuda:0',
        # max_epochs=train_config['epochs'],
        logger=make_logger(config_dict),
        enable_progress_bar=True,
    )

    prediction = trainer.predict(model, dataloaders=DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, drop_last=True))
    print(prediction)


if __name__ == '__main__':
    evaluate()



