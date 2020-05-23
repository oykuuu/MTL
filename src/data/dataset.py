""" #TODO
"""

import os
import numpy as np
import torchvision
import torch
import pandas as pd
from typing import Tuple

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

import pdb

class CAPTCHA(Dataset):
    """CAPTCHA dataset."""

    def __init__(self, data_root, train=True, transform=None) -> None:
        """
        Parameters
        ----------
            data_root : str 
                Root of the data directory.
            train : boolean, optional
                Whether to load train or test data, default train.
            transform : callable, optional
                Optional transform to be applied to the image
        """

        self.data_root = data_root
        self.transform = transform
        self.train = train

        self._init_dataset()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_files = self.samples[idx]
        image = np.array([Image.open(image_file) for image_file in image_files])
        label = self.labels[idx]

        if not self.transform:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ]) 

        image = self.transform(image)

        return (image, label)

    def _init_dataset(self) -> None:
        self.samples = [os.path.join(self.data_root, filename) for filename in os.listdir(self.data_root)]
        self.labels = [filename.split('.')[0] for filename in os.listdir(self.data_root)]
        self.captcha_chars = {letter for captcha in labels for letter in captcha}


if __name__ == "__main__":

    transform = transforms.Compose([
                    transforms.Scale(28),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = CAPTCHA(data_root='~/Documents/code/mtl_data', train=True)
    test_dataset = CAPTCHA(data_root='~/Documents/code/mtl_data', train=False)

    print(len(train_dataset))
    print(train_dataset[10])

    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=50, shuffle=True, num_workers=2)
   
    print(next(iter(train_dataloader)))

    pdb.set_trace()

