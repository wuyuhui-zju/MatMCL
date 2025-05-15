import os
from PIL import Image
import torch
import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms


def random_index(lst, value, is_random=True, direction=None):
    indices = [i for i, x in enumerate(lst) if x == value]
    if not indices:
        return None

    if is_random:
        return random.choice(indices)
    else:
        if direction == "ori":
            return indices[0]
        elif direction == "ver":
            return indices[1]
        else:
            raise ValueError


class MultiModalDataset(Dataset):
    def __init__(self, root_path, dataset, scaler):
        super().__init__()
        # tabular modal
        tabular_data_path = os.path.join(root_path, f"table/{dataset}/{dataset}.csv")
        df = pd.read_csv(tabular_data_path)
        tabular_data = df.values[:, 1:]

        tabular_data_cont = tabular_data[:, :-1]
        tabular_data_cont = scaler.transform(tabular_data_cont)
        tabular_data = np.concatenate([tabular_data_cont, tabular_data[:, -1][:, np.newaxis]], axis=1)

        self.tabular_data = torch.tensor(tabular_data, dtype=torch.float32)
        self.ID_list = df["ID"].astype(int).tolist()

        # image modal
        self.img_data_path = os.path.join(root_path, f"images/{dataset}")
        self.img_fns = os.listdir(self.img_data_path)

        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        self.scaler = scaler

    def __getitem__(self, item):
        # Load and transform the image
        fn = self.img_fns[item]
        image = Image.open(os.path.join(self.img_data_path, fn))

        sID = int(fn.split("_")[0])
        table_idx = random_index(self.ID_list, sID, is_random=False, direction="ori")
        x_table = self.tabular_data[table_idx]
        x_image = self.transform(image)

        return x_table, x_image

    def __len__(self):
        return len(self.img_fns)


class ImageDataset(Dataset):
    def __init__(self, root_path, dataset):
        super().__init__()

        # image modal
        self.img_data_path = os.path.join(root_path, "images", dataset)
        self.img_fns = os.listdir(self.img_data_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
        ])

    def __getitem__(self, item):
        fn = self.img_fns[item]
        image = Image.open(os.path.join(self.img_data_path, fn))
        x_image = self.transform(image)
        return x_image

    def __len__(self):
        return len(self.img_fns)
