from typing import Iterable
from lightning.pytorch.loggers import Logger, TensorBoardLogger

def make_logger(config: Iterable[dict] | dict) -> list[Logger]:
    # logger_cfgs = config['logger']
    loggers = list()
    if isinstance(config, list):
        loggers = [make_single_logger(config) for config in config]
    elif isinstance(config, dict):
        loggers = [make_single_logger(config)]
    return loggers

def make_single_logger(config: dict) -> Logger:
    logger_cfg = config['logger']

    if logger_cfg['type'] == "tensorboard":
        logger = TensorBoardLogger(
            save_dir=logger_cfg.get("save_dir", "/root/tf-logs"),
            name=logger_cfg.get("name", "temp_exp"),
            version=logger_cfg.get("version"),
        )
        hparams = dict(
            # pretrain_weight = config['model']['pretrain_weight'] if 'pretrain_weight' in config['model'] else None,
            lr = config['train']['lr'],
            epochs = config['train']['epochs'],

            batch_size =  config['model']['batch_size'],
            window_size = config['model']['window_size'],
            feature_number = config['model']['feature_number'],
            N = config['model']['N'],
            d_model_Encoder = config['model']['d_model_Encoder'],
            d_model_Decoder = config['model']['d_model_Decoder'],
            d_ff_Encoder = config['model']['d_ff_Encoder'],
            d_ff_Decoder = config['model']['d_ff_Decoder'],
            h = config['model']['h'], #heads_num
            dropout = config['model']['dropout'],
            local_context_length = config['model']['local_context_length'],

        )
        logger.log_hyperparams(hparams)  #log the essential hparams for benchmarking
        return logger
    else:
        return TensorBoardLogger("logs")