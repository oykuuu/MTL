import torch
import torchvision
import numpy as np

from typing import Tuple

from torch.utils.data import DataLoader, random_split

import data.dataset


class CustomRandomCrop(object):
    def __init__(self, output_width):
        """Crop randomly the image in a sample. Crop only on the sides.

        Parameters
        ----------
        output_width
            int, Desired output width
        """
        assert isinstance(output_width, (int))
        self.output_width = output_width

    def __call__(self, image):
        """ Crop from the right or left randomly. Sometimes skip
        transform.

        Parameters
        ----------
        image
            PIL.Image, image to be transformed
        Returns
        -------
        image
            PIL.Image, transformed image
        """
        if np.random.random() > 0.5:
            # half of the time skip transform
            return image

        w, h = image.size
        crop_by = np.random.randint(0, w - self.output_width)
        # crop from the left (right) half of the time
        if np.random.random() > 0.5:
            image = image.crop((crop_by, 0, w, h))
        else:
            image = image.crop((0, 0, w - crop_by, h))

        return image


def get_dataloaders(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    random_seed: int,
    shuffle: bool,
    val_split: float = 0.2,
    test_split: float = 0.2,
    num_workers: int = 4,
    pin_memory: bool = False,
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:

    """
    Function for loading and splitting data for training, validation, test for CAPTCHA

    Parameters
    ----------
    dataset
        torch.utils.data.Dataset, CAPTCHA dataset object.
    batch_size
        int, Number of samples per batch.
    random_seed
        int, randomness seed.
    shuffle
        boolean, flag for shuffling samples in the dataloader.
    val_split
        float, ratio for validation data.
    test_split
        float, ratio for test data.       
    num_workers
        int, number of processes for data loading.
    pin_memory
        boolean, copy tensors into CUDA pinned memory. Should be set if using GPUs

    Returns
    -------
    train_loader
        torch.utils.data.DataLoader, training set iterator
    val_loader
        torch.utils.data.DataLoader, validation set iterator
    test_loader
        torch.utils.data.DataLoader, test set iterator
    """
    # for reproducible data splits
    torch.manual_seed(random_seed)

    # establish train-val-test splits
    n_data = len(dataset)
    n_test = int(np.floor(test_split * n_data))
    n_val = int(np.floor(val_split * n_data))
    train_val, test_dataset = random_split(dataset, [n_data - n_test, n_test])

    train_dataset, val_dataset = random_split(
        train_val, [len(train_val) - n_val, n_val]
    )

    # initialize all dataloaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # return the random seed to its original value
    torch.manual_seed(torch.initial_seed())

    return train_loader, val_loader, test_loader
