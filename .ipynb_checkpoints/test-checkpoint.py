
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
    # dataloaders = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0, drop_last=False)
    prediction = trainer.predict(model, dataloaders=DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0, drop_last=False))
    # print(prediction)
    # prediction = pd.DataFrame(data=[np.arange(len(prediction)),prediction], columns=["ID", 'prediction'])
    prediction = torch.cat(prediction, dim=0)
    pred = prediction[:, -1, 0]
    predictions = pd.DataFrame(data=pred, columns=['prediction'])
    
    predictions.to_csv(output_path, index=True)


if __name__ == '__main__':
    #Best_ckpt = train()
    evaluate('/root/tf-logs/temp_exp/version_0/checkpoints/best-IC-epoch=00-IC=0.0321.ckpt', '/root/output2.csv')

