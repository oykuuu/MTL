""" TODO
"""

import os
import json
import copy
import time
import argparse
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms

from data.dataloader import get_dataloaders, CustomRandomCrop
from data.dataset import CAPTCHA_MultiTask, CAPTCHA_SingleTask
from models.multi_task import MultiTaskFinalLayer
from models.single_task import SingleTaskFinalLayer
from utils.utils import get_loss, get_max_from_list, plot_grad_flow

import pdb


def train_model(model, dataloaders, criterion, alphas, optimizer, num_epochs=10, device="cpu", num_tasks=1):
    """
    Trains the Deep Learning model. Keeps track of the best model according to the
    holdout validation set.

    Parameters
    ----------
    model
        model, Object Classification model to be trained
    dataloaders
        dict, dictionary of PyTorch DataLoader for train and validation set
    criterion
        nn.criterion, loss function used in optimization
    alphas
        list, weights in the linear combination of loss functions
    optimizer
        nn.optim, method for converging to the optimal value
    num_epochs
        int, number of epochs to train for
    device
        string, set to 'gpu' if available
    num_tasks
        int, number of tasks

    Returns
    -------
    model
        model, trained model
    val_acc_history
        list, list of validation accuracy at different timesteps
    """
    since = time.time()

    val_acc_history = []
    best_model_params = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        epoch_message = "\nEpoch {}/{}".format(epoch + 1, num_epochs)
        print(epoch_message)
        print("-" * len(epoch_message))

        # in every epoch, train once then evaluate with validation set
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # run through train/validation set examples.
            # If train, update gradients. If validation, run evaluation
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = get_loss(criterion, outputs, labels, alphas)
                    preds = get_max_from_list(outputs)

                    if phase == "train":
                        loss.backward()
                        plot_grad_flow(model.named_parameters())
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

                if phase == "valid":
                    val_acc_history.append(epoch_acc)
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_params = copy.deepcopy(model.state_dict())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / (len(dataloaders[phase].dataset) * num_tasks)
            print(
                "{} - Loss: {:.4f} \t Acc: {:.4f}".format(phase, epoch_loss, epoch_acc)
            )
    time_elapsed = time.time() - since

    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best validation accuracy: {:.4f}\n\n".format(best_acc))
    print("-" * 10)

    model.load_state_dict(best_model_params)
    return model, val_acc_history


def construct_model(num_classes, task_type, num_tasks, device="cpu"):
    """
    Obtain the pretrained Resnet-18 model and replace the last layer
    accordingly for single or multi task learnings.

    Parameters
    ----------
    num_classes
        int, number of classes
    task_type
        string, determines how the last layer of ResNet is constructed
    num_tasks
        int, numberp of tasks
    device
        string, set to 'gpu' if available

    Returns
    -------
    model
        model, pretrained Resnet-18 model
    params_to_update
        list, list of the parameters to update (last layer only)
    """
    model = models.resnet18(pretrained=True, progress=True)
    for param in model.parameters():
        param.requires_grad = False

    # by default last layers will be initialized with requires_grad=True
    if task_type == "single":
        model.fc = SingleTaskFinalLayer(num_classes)
    elif task_type == "multi":
        model.fc = nn.Linear(model.fc.in_features, 256)
        model = MultiTaskFinalLayer(model, num_classes, num_tasks)
    else:
        raise ValueError("Not a valid task type.")
    model = model.to(device)

    params_to_update = [
        param for name, param in model.named_parameters() if param.requires_grad
    ]

    return model, params_to_update


def get_test_predictions(model, test_dataloader, encoder, num_tasks):
    """
    Obtain predictions for the test set

    Parameters
    ----------
    model
        model, trained model
    test_dataloader
        DataLoader, dataloader of the test set
    encoder
        LabelEncoder, encoder used to turn CAPTCHA characters into integer labels.

    Returns
    -------
    captcha_chars
        list, predicted CAPTCHA characters
    """
    predicted_chars = []
    model.eval()
    running_corrects = 0
    for inputs, labels in test_dataloader:
        outputs = model(inputs)
        preds = get_max_from_list(outputs)
        captcha_chars = encoder.inverse_transform(np.array(preds).ravel())
        _ = [predicted_chars.append(c) for c in captcha_chars]
        running_corrects += torch.sum(preds == labels.squeeze(1))
    
    acc = running_corrects.double() / (len(test_dataloader.dataset) * num_tasks)

    print('\nTest accuracy: {}'.format(acc))
    return predicted_chars, acc


def main(config_path):
    # config = json.load(open(config_path, "r"))

    # hyperparameters
    task_type = 'multi'
    data_root='/Users/oh761139/Documents/code/mtl_data/CAPTCHA/samples'
    batch_size = 16
    random_seed = 42
    shuffle = True
    val_split = 0.1
    test_split = 0.1
    learning_rate = 1e-3
    momentum = 0.9
    num_epochs = 10
    ####


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device {}\n".format(device))

    # transformations for data augmentation
    transform = transforms.Compose([
        CustomRandomCrop(180),
        transforms.RandomRotation(10, resample=Image.BILINEAR),
        transforms.Resize([50,200]),
        transforms.ToTensor(),
    ])

    # assign datasets
    if task_type.lower() == 'single':
        dataset = CAPTCHA_SingleTask(data_root, transform=transform, char_place=0)
    elif task_type.lower() == 'multi':
        dataset = CAPTCHA_MultiTask(data_root, transform=transform)
    else:
        raise ValueError("Not a valid task type.")
    num_classes = dataset.get_num_classes()
    num_tasks = dataset.get_num_tasks()
    encoder = dataset.get_encoder()

    # assign loss function
    criterion = [nn.CrossEntropyLoss(reduction='sum')] * num_tasks
    alphas = np.ones([num_tasks]) / num_tasks

    # get dataloaders
    train_dataloader, valid_dataloader, test_dataloader = get_dataloaders(
        dataset,
        batch_size,
        random_seed,
        shuffle,
        val_split,
        test_split,
    ) 

    dataloaders = {}
    dataloaders["train"] = train_dataloader
    dataloaders["valid"] = valid_dataloader

    # load in Resnet-18 model
    model, params_to_update = construct_model(num_classes, task_type, num_tasks, device)

    #optimizer = optim.SGD(params_to_update, lr=learning_rate, momentum=momentum)
    optimizer = optim.Adam(params_to_update, lr=learning_rate)

    # train model
    model, val_acc_history = train_model(
        model, dataloaders, criterion, alphas, optimizer, num_epochs=num_epochs, device=device, num_tasks=num_tasks
    )

    # plot the gradients for sanity check
    plt.tight_layout()
    plt.show()

    # get predictions
    test_chars, test_acc = get_test_predictions(model, test_dataloader, encoder, num_tasks)

    pdb.set_trace()


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        "-c", "--config", help="filepath to config json", default="./config.json"
    )
    ARGS = PARSER.parse_args()
    CONFIGPATH = ARGS.config
    main(CONFIGPATH)