import os
from PIL import Image
import torch
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torchvision.transforms.functional as F


class Rotate90:
    def __call__(self, img):
        return F.rotate(img, 90)


class MultiModalDataset(Dataset):
    def __init__(self, root_path="../../datasets", dataset_type="train", split_name="sgpt", image_size=224, scaler=None):
        super().__init__()
        self.scaler = scaler
        # tabular modal
        tabular_data_path = os.path.join(root_path, "table", split_name, f"{dataset_type}.csv")
        df = pd.read_csv(tabular_data_path)
        tabular_data = df.values[:, 1:8]

        if scaler is None:
            scaler = StandardScaler()
            tabular_data_cont = tabular_data[:, :-1]
            tabular_data_cont = scaler.fit_transform(tabular_data_cont)
            tabular_data = np.concatenate([tabular_data_cont, tabular_data[:, -1][:, np.newaxis]], axis=1)
            self.scaler = scaler
        else:
            tabular_data_cont = tabular_data[:, :-1]
            tabular_data_cont = self.scaler.transform(tabular_data_cont)
            tabular_data = np.concatenate([tabular_data_cont, tabular_data[:, -1][:, np.newaxis]], axis=1)

        self.tabular_data = torch.tensor(tabular_data, dtype=torch.float32)
        self.ID_list = df["ID"].astype(int).tolist()

        # image modal
        self.img_data_path = os.path.join(root_path, "images/preprocessed")

        # Define image transformations
        self.transform_ori = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize the image to the desired size
            transforms.RandomHorizontalFlip(p=0.1),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ToTensor(),  # Convert the image to a tensor
        ])

        self.transform_ver = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize the image to the desired size
            transforms.RandomHorizontalFlip(p=0.1),
            transforms.RandomVerticalFlip(p=0.1),
            Rotate90(),
            transforms.ToTensor(),  # Convert the image to a tensor
        ])

    def __getitem__(self, item):
        x_tabular = self.tabular_data[item]
        idx = self.ID_list[item]
        img_path = os.path.join(self.img_data_path, str(idx))
        fn = random.choice(os.listdir(img_path))
        img_full_path = os.path.join(img_path, fn)

        # Load and transform the image
        image = Image.open(img_full_path)

        direction = int(x_tabular[-1])
        if bool(x_tabular[-1] == 1):
            x_image = self.transform_ori(image)
        else:
            x_image = self.transform_ver(image)

        return idx, direction, x_tabular, x_image

    def __len__(self):
        return len(self.ID_list)


if __name__ == "__main__":
    pass