import torch
from torch.utils.data import Dataset


def process_parquet():
    pass

class kaggle_dataset(Dataset):
    def __init__(self):
        super(kaggle_dataset).__init__()

        #data processing for parquet
        process_parquet()

    def __len__(self):
        return 10

    def __getitem__(self, item):
        return item