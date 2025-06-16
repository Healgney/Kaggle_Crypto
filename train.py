
import pandas as pd
import lightning as pl
import torch
from torch.utils.data import DataLoader
import numpy as np
import yaml

from HeLU.model.HeLU_model import HeLU_Crypto
from HeLU.callback import logger_callback
from HeLU.dataset.crypto_dataset import ParquetDataset
from HeLU.logger.logger_factory import make_logger
from lightning.pytorch.callbacks import ModelCheckpoint


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
    dataset_train = ParquetDataset('/root/drw-crypto-market-prediction/train.parquet', model_config['window_size'], mode='train')
    dataset_val = ParquetDataset('/root/drw-crypto-market-prediction/train.parquet', model_config['window_size'], mode='val')

    model = HeLU_Crypto(model_config)

    if "optimizer" in config_dict['train'].keys():
        model.set_optimizer_config(**config_dict['train']["optimizer"])
    checkpoint_callback = ModelCheckpoint(
        monitor="IC",                
        mode="max",                 
        save_top_k=1,
        filename="best-IC-{epoch:02d}-{IC:.4f}",
        save_weights_only=False,
    )
    trainer = pl.Trainer(
        accelerator='cuda',
        callbacks=[checkpoint_callback],
        # devices='cuda:0',
        max_epochs=train_config['epochs'],
        logger=make_logger(config_dict),
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    trainer.fit(model, train_dataloaders = DataLoader(dataset_train, batch_size=106, shuffle=True, num_workers=0, drop_last=True),
                val_dataloaders = DataLoader(dataset_val, batch_size=106, shuffle=True, num_workers=0, drop_last=True))
    return checkpoint_callback.best_model_path
def evaluate(path, output_path:str=''):
    config_dict = configure()
    model_config = config_dict['model']
    train_config = config_dict['train']
    dataset = ParquetDataset('/root/drw-crypto-market-prediction/test.parquet', model_config['window_size'], mode = 'test')

    model = HeLU_Crypto.load_from_checkpoint(path)

    trainer = pl.Trainer(
        accelerator='cuda',
        # devices='cuda:0',
        # max_epochs=train_config['epochs'],
        logger=make_logger(config_dict),
        enable_progress_bar=True,
    )

    prediction = trainer.predict(model, dataloaders=DataLoader(dataset, batch_size=106, shuffle=False, num_workers=0, drop_last=False))
    # print(prediction)
    # prediction = pd.DataFrame(data=[np.arange(len(prediction)),prediction], columns=["ID", 'prediction'])
    # prediction = torch.cat(prediction).squeeze().cpu().numpy()

    prediction = pd.DataFrame(data=prediction, columns=['prediction'])
    
    prediction.to_csv(output_path, index=True)


if __name__ == '__main__':
    Best_ckpt = train()
    evaluate(Best_ckpt, '/root/output.csv')

