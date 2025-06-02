from Model.model_factory import make_model
from Model.Run import Crypto
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

    model = make_model(*model_config)
    crypto = Crypto(model)

    trainer = pl.Trainer(train_config)

    trainer.fit(model)

def evaluate():
    config_dict = configure()

    model_config = config_dict['model']
    train_config = config_dict['train']

    model = make_model(*model_config)
    crypto = Crypto(model)

    trainer = pl.Trainer(train_config)

    trainer.fit(model)


if __name__ == '__main__':
    train()