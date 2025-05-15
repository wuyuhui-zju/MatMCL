import os
import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset


class TableDataset(Dataset):
    def __init__(self, split, dataset_type, root_path="../datasets", scaler=None):
        super().__init__()
        tabular_data_path = os.path.join(root_path, f"table/{split}/{dataset_type}.csv")
        df = pd.read_csv(tabular_data_path)
        tabular_data = df.values[:, 1:8]
        tabular_data_cont = scaler.transform(tabular_data[:, :-1])
        tabular_data = np.concatenate([tabular_data_cont, tabular_data[:, -1][:, np.newaxis]], axis=1)
        self.tabular_data = torch.tensor(tabular_data, dtype=torch.float32)

        labels = df.values[:, 8:12]
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.mean = None
        self.std = None
        self.set_mean_and_std()
        self.n_tasks = labels.shape[1]

    def __getitem__(self, item):
        return self.tabular_data[item], self.labels[item]

    def __len__(self):
        return self.labels.size()[0]

    def set_mean_and_std(self, mean=None, std=None):
        if mean is None:
            mean = torch.from_numpy(np.nanmean(self.labels.numpy(), axis=0))
        if std is None:
            std = torch.from_numpy(np.nanstd(self.labels.numpy(), axis=0))
        self.mean = mean
        self.std = std


if __name__ == "__main__":
    pass
