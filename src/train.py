""" TODO
"""

import os
import json
import copy
import time
import argparse
import numpy as np
import cv2

from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim

from data.dataloader import get_dataloaders
from data.dataset import CAPTCHA_MultiTask, CAPTCHA_SingleTask

import pdb

def train_model(model, dataloaders, criterion, optimizer, num_epochs=10, device="cpu"):
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
    optimizer
        nn.optim, method for converging to the optimal value
    num_epochs
        int, number of epochs to train for
    device
        string, set to 'gpu' if available

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
        print("\nEpoch {}/{}".format(epoch + 1, num_epochs))
        print("----------")

        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.squeeze(1))
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                if phase == "valid":
                    val_acc_history.append(epoch_acc)
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_params = copy.deepcopy(model.state_dict())
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

class SingleTaskFinalLayer(nn.Module):
    """ In the case of single task learning, determines the final layer of ResNet.
    """
    def __init__(self, num_classes):
        """ Initialize a single linear layer.

        Parameters
        ----------
        num_classes
            int, number of classes
        Returns
        -------
        None
        """
        super(SingleTaskFinalLayer, self).__init__()
        self.layer = nn.Linear(512, num_classes)
        
    def forward(self, x):
        """ Forward pass over final layer.

        Parameters
        ----------
        x
            tensor, output of the ResNet just before the final layer.
        Returns
        -------
        self.layer(x)
            tensor, class probabilities
        """
        return self.layer(x)


class MultiTaskFinalLayer(nn.Module):
    """ In the case of multi-task learning, determines the final layer of ResNet.
    """
    def __init__(self, num_classes, num_tasks):
        """ Initialize final linear layers for each task.

        Parameters
        ----------
        num_classes
            int, number of classes
        Returns
        -------
        None
        """
        super(MultiTaskFinalLayer, self).__init__()
        self.layers = [nn.Linear(512, num_classes) for _ in range(num_tasks)]
        
    def forward(self, x):
        """ Forward pass over final layer.

        Parameters
        ----------
        x
            tensor, output of the ResNet just before the final layer.
        Returns
        -------
        outputs
            list, list of class probabilities for each task
        """
        outputs = [task_layer(x) for task_layer in self.layers]
        return outputs


def get_resnet(num_classes, task_type, num_tasks, device="cpu"):
    """
    Obtain the pretrained Resnet-18 model and allow for the training of
    the last layer only.

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
        model.fc = MultiTaskFinalLayer(num_classes, num_tasks)
    else:
        raise ValueError("Not a valid task type.")

    model = model.to(device)

    params_to_update = [
        param for name, param in model.named_parameters() if param.requires_grad
    ]

    return model, params_to_update


def get_team_predictions(model, team_dataloader, encoder):
    """
    Given a model and a dataloader, predicts the players present in a team.

    Parameters
    ----------
    model
        model, trained model
    team_dataloader
        DataLoader, dataloader containing images of only one team
    encoder
        LabelEncoder, encoder used to turn champion names into integers

    Returns
    -------
    names_predicted
        list, names of the predicted champions
    """
    predictions = []
    model.eval()
    for inputs, _ in team_dataloader:
        preds = nn.LogSoftmax(dim=1)(model(inputs)).detach().numpy()
        pred = np.argmax(preds)
        predictions.append(pred)

    names_predicted = encoder.inverse_transform(predictions)
    return list(names_predicted)


def main(config_path):
    # config = json.load(open(config_path, "r"))
    # game_img_path = config["paths"]["game_image_path"]
    # champion_folder = config["paths"]["champion_folder"]

    # num_classes = len(os.listdir(champion_folder))
    # num_epochs = config["dl"]["epochs"]
    # learning_rate = config["dl"]["learning_rate"]
    # momentum = config["dl"]["momentum"]
    # batch_size = config["dl"]["batch_size"]

    # hyperparameters
    task_type = 'single'
    data_root='/Users/oh761139/Documents/code/mtl_data/CAPTCHA/samples'
    batch_size = 16
    random_seed = 42
    shuffle = True
    val_split = 0.2
    test_split = 0.2
    learning_rate = 1e-4
    momentum = 0.9
    num_epochs = 2
    ####


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # assign datasets
    if task_type.lower() == 'single':
        dataset = CAPTCHA_SingleTask(data_root, transform=None, char_place=0)
    elif task_type.lower() == 'multi':
        dataset = CAPTCHA_MultiTask(data_root, transform=None)
    else:
        raise ValueError("Not a valid task type.")
    num_classes = dataset.get_num_classes()
    num_tasks = dataset.get_num_tasks()
        

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
    model, params_to_update = get_resnet(num_classes, task_type, num_tasks, device)

    optimizer = optim.SGD(params_to_update, lr=learning_rate, momentum=momentum)
    #optimizer = optim.Adam(params_to_update, lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # train model
    model, val_acc_history = train_model(
        model, dataloaders, criterion, optimizer, num_epochs=num_epochs, device=device
    )
    pdb.set_trace()
    # get team predictions
    left_dataloader, left_true_labels = get_valid_dataloader(
        encoder, data_path=game_img_path, team="left", batch_size=1, shuffle=False
    )
    right_dataloader, right_true_labels = get_valid_dataloader(
        encoder, data_path=game_img_path, team="right", batch_size=1, shuffle=False
    )
    left_players = get_team_predictions(model, left_dataloader, encoder)
    right_players = get_team_predictions(model, right_dataloader, encoder)

    print("Left team members are predicted as: \n {}\n".format(left_players))
    print("Actual feft team members are: \n {}\n\n".format(left_true_labels))
    print("Right team members are predicted as: \n {}\n".format(right_players))
    print("Actual right team members are: \n {}\n".format(right_true_labels))


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        "-c", "--config", help="filepath to config json", default="./config.json"
    )
    ARGS = PARSER.parse_args()
    CONFIGPATH = ARGS.config
    main(CONFIGPATH)