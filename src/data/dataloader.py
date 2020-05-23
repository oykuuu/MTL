import torch
import torchvision
import numpy as np

from typing import Tuple

from torch.utils.data import DataLoader, random_split

import data.dataset


def get_dataloaders(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    random_seed: int,
    shuffle: bool,
    val_split: float = 0.2,
    test_split: float = 0.2,
    num_workers: int = 4,
    pin_memory: bool = False,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

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
    train_val, test_dataset = random_split(dataset, [n_data - n_test, n_test])

    n_val = int(np.floor(val_split * len(train_val)))
    train_dataset, val_dataset = random_split(train_val, [len(train_val) - n_val, n_val])

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

