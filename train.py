from HeLU.model.model_factory import make_model
from HeLU.model.HeLU_model import HeLU_Crypto
from HeLU.dataset.crypto_dataset import ParquetDataset

from HeLU.callback import logger_callback
from torch.utils.data import DataLoader
import lightning as pl
import yaml

def configure(config_path = 'config/config.yaml'):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return config_dict


def train():
    config_dict = configure()

    model_config = config_dict['model']
    train_config = config_dict['train']

    dataset = ParquetDataset('/root/autodl-tmp/train.parquet')
    train_dataset = DataLoader(dataset, batch_size=1, shuffle=True)

    model = HeLU_Crypto(model_config)

    trainer = pl.Trainer(
        devices='mps',
        max_epochs=train_config['epochs'],
        progress_bar_refresh_rate=0,
        logger=True,
        enable_progress_bar=True
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_dataset
    )

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
    train()