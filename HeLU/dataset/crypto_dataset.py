import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import pandas as pd
from glob import glob


class ParquetDataset(Dataset):
    def __init__(self, parquet_file, label_col='label', device='cpu', remove_nan=True):
        self.device = device
        df = pd.read_parquet(parquet_file)

        print(len(list(df.columns)))
        cols_to_drop = [f"X{i}" for i in range(697, 718)]
        df.drop(columns=cols_to_drop, inplace=True)
        print(len(list(df.columns)))

        df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)

        if remove_nan:
            df.dropna(inplace=True)

        self.labels = torch.tensor(df[label_col].values, dtype=torch.float32)

        feature_df = df.drop(columns=[label_col])
        self.features = torch.tensor(feature_df.values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.features[idx].to(self.device)
        y = self.labels[idx].to(self.device)
        return x, y


if __name__ == '__main__':

    dataset = ParquetDataset('/root/autodl-tmp/train.parquet')

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for batch_x, batch_y in dataloader:
        print(batch_x.shape, batch_y.shape)
        break