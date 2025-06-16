import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import pandas as pd
from glob import glob


class ParquetDataset(Dataset):
    def __init__(self, parquet_file, window_size, label_col='label', device='cuda', mode='train', train_ratio=0.8,
                 remove_nan=True):
        super().__init__()
        self.device = device
        self.mode = mode

        df = pd.read_parquet(parquet_file).iloc[:6 * window_size]
        total_len = len(df)
        train_len = int(total_len * train_ratio)
        if self.mode == 'train':
            df = df.iloc[:train_len]
        elif self.mode == 'val':
            df = df.iloc[train_len:]
        elif self.mode == 'test':
            feature_df = df
            self.features = torch.tensor(feature_df.values, dtype=torch.float32)
            self.labels = None
            # 非重叠窗口起始位置列表
            self.indices = list(range(0, len(self.features) - window_size + 1, window_size))
            return
        else:
            raise ValueError("mode must be 'train', 'val', 'test', or 'infer'")

        self.window_size = window_size
        cols_to_drop = [f"X{i}" for i in range(697, 718)]
        df.drop(columns=cols_to_drop, inplace=True)

        df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)

        if remove_nan:
            df.dropna(inplace=True)

        self.labels = torch.tensor(df[label_col].values, dtype=torch.float32)
        feature_df = df.drop(columns=[label_col])
        self.features = torch.tensor(feature_df.values, dtype=torch.float32)
        pad_w = self.window_size
        self.features = F.pad(self.features, (0, 0, 0, pad_w + 1))
        self.labels = F.pad(self.labels, (0, pad_w + 1))

        # S = len(self.labels)# - self.window_size + 1
        # B = 128
        # padding_sample = (B - (S % B)) % B

        # if padding_sample > 0:
        #     self.features = F.pad(self.features, (0, 0, 0, padding_sample-1))
        #     # print(f"padding sample: {self.features.size()}")
        #     self.labels = F.pad(self.labels, (0, padding_sample-1))

        #     # print(f"padding labels: {self.labels.size()}")

    def __len__(self):
        if self.mode == 'train':
            return len(self.labels) - self.window_size + 1
        elif self.mode == 'val':
            return len(self.labels) - self.window_size + 1
        elif self.mode == 'test':
            return len(self.indices)
        else:
            raise ValueError("mode must be 'train' or 'val'")

    def __getitem__(self, idx):
        if self.mode == 'test':
            start_idx = self.indices[idx]
            x = self.features[start_idx:start_idx + self.window_size, :].to(self.device)
            return x
        else:
            x = self.features[idx:idx + self.window_size, :].to(self.device)
            y = self.labels[idx:idx + self.window_size].unsqueeze(-1).to(self.device)
            return x, y

