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
from sklearn.preprocessing import LabelEncoder

import pdb


class CAPTCHA_MultiTask(Dataset):
    """CAPTCHA dataset."""

    def __init__(self, data_root, transform=None) -> None:
        """
        Parameters
        ----------
            data_root
                str, Root of the data directory.
            transform
                callable, optional transform to be applied to the image
        """

        self.data_root = data_root
        self.transform = transform
        self.encoder = LabelEncoder()
        self.image_paths = []

        self._init_dataset()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Returns item at index idx.
        Parameters
        ----------
        idx
            int, sample index number
        Returns
        -------
        image
            tensor, image at idx
        label
            tensor, multi-labels of the image (5 characters long)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if not self.transform:
            self.transform = transforms.Compose([transforms.ToTensor()])

        path, label = self.image_paths[idx]
        # encoded_label = np.array([self.encoder.transform([letter]) for letter in label])
        encoded_label = self.encoder.transform(list(label))
        image = Image.open(path).convert("RGB")
        image = self.transform(image)

        return (image, encoded_label)

    def _init_dataset(self):
        """
        Dataset initalizer. Looks into data_root folder and collects
        unique characters in CAPTCHAs.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        all_labels = []
        for name in os.listdir(self.data_root):
            label, extension = name.split(".")
            if not (extension == "png" or extension == "jpg"):
                continue
            all_labels.append(label)
            self.image_paths += [(os.path.join(self.data_root, name), label)]

        captcha_chars = {letter for captcha in all_labels for letter in captcha}
        self.encoder = self.encoder.fit(list(captcha_chars))
        self.num_classes = len(captcha_chars)
        self.num_tasks = len(all_labels[0])

    def get_num_tasks(self):
        return self.num_tasks

    def get_num_classes(self):
        return self.num_classes

    def get_encoder(self):
        return self.encoder


class CAPTCHA_SingleTask(Dataset):
    """CAPTCHA dataset."""

    def __init__(self, data_root, transform=None, char_place=0) -> None:
        """
        Parameters
        ----------
            data_root
                str, Root of the data directory.
            transform
                callable, optional transform to be applied to the image
            char_place
                int, single char to predict
        """

        self.data_root = data_root
        self.transform = transform
        self.char_place = char_place
        self.encoder = LabelEncoder()
        self.image_paths = []

        self._init_dataset()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Returns item at index idx.
        Parameters
        ----------
        idx
            int, sample index number
        Returns
        -------
        image
            tensor, image at idx
        label
            tensor, multi-labels of the image (5 characters long)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if not self.transform:
            self.transform = transforms.Compose([transforms.ToTensor()])

        path, label = self.image_paths[idx]
        label = label[self.char_place]
        encoded_label = self.encoder.transform([label])[0]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)

        return (image, encoded_label)

    def _init_dataset(self):
        """
        Dataset initalizer. Looks into data_root folder and collects
        unique characters in CAPTCHAs.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        all_labels = []
        for name in os.listdir(self.data_root):
            label, extension = name.split(".")
            if not (extension == "png" or extension == "jpg"):
                continue
            all_labels.append(label)
            self.image_paths += [(os.path.join(self.data_root, name), label)]

        captcha_chars = {letter for captcha in all_labels for letter in captcha}
        self.encoder = self.encoder.fit(list(captcha_chars))
        self.num_classes = len(captcha_chars)
        self.num_tasks = len(all_labels[0])

    def get_num_tasks(self):
        return self.num_tasks

    def get_num_classes(self):
        return self.num_classes

    def get_encoder(self):
        return self.encoder


if __name__ == "__main__":
    data_root = "~/Documents/code/mtl_data/CAPTCHA/samples"
    dataset = CAPTCHA_MultiTask(data_root, transform=None)

    print(dataset[10])
