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

import pdb

def plot_grad_flow(named_parameters):
    """ Sanity checks of gradient updates in backprop.

    Parameters
    ----------
    named_parameters
        parameters to be updated

    Returns
    -------
    None
    """

    ave_grads = []
    max_grads = []
    layers = []

    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation=30)
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    


def get_max_from_list(outputs):
    if isinstance(outputs, list):
        flat_preds = torch.stack([torch.max(task_outputs, 1)[1] for task_outputs in outputs])
        return flat_preds.T
    if outputs.dim() == 2:
        _, flat_preds = torch.max(outputs, 1)
    elif outputs.dim() == 3:
        # if outputs dimension is 3, this was a multi-task learning
        flat_preds = torch.stack([torch.max(task_outputs, 1)[1] for task_outputs in outputs])
    else:
        raise ValueError("Dimension of the output must be 2 or 3.")

    return flat_preds


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
                    #loss = criterion(outputs, labels)

                    loss1 = criterion[0](outputs[:, 0, :], labels[:, 0])
                    loss2 = criterion[1](outputs[:, 1, :], labels[:, 1])
                    loss3 = criterion[2](outputs[:, 2, :], labels[:, 2])
                    loss4 = criterion[3](outputs[:, 3, :], labels[:, 3])
                    loss5 = criterion[4](outputs[:, 4, :], labels[:, 4])
                    loss = loss1 + loss2 + loss3 + loss4 + loss5


                    
                    preds = get_max_from_list(outputs)

                    if phase == "train":
                        loss.backward()
                        plot_grad_flow(model.named_parameters())
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels) # torch.tensor(0)

                if phase == "valid":
                    val_acc_history.append(epoch_acc)
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_params = copy.deepcopy(model.state_dict())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
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
    def __init__(self, resnet_base, num_classes, num_tasks):
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
        self.resnet_base = resnet_base

        # resnet.fc was replaced with Linear (512, 256)
        # so self.bn1 is for that linear layer
        self.bn1 = nn.BatchNorm1d(256, eps = 2e-1)
        self.x2 =  nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x2.weight)
        self.bn2 = nn.BatchNorm1d(256, eps = 2e-1)
        
        
        self.task1 = nn.Linear(256, num_classes)
        nn.init.xavier_normal_(self.task1.weight)
        self.task2 = nn.Linear(256, num_classes)
        nn.init.xavier_normal_(self.task2.weight)
        self.task3 = nn.Linear(256, num_classes)
        nn.init.xavier_normal_(self.task3.weight)
        self.task4 = nn.Linear(256, num_classes)
        nn.init.xavier_normal_(self.task4.weight)
        self.task5 = nn.Linear(256, num_classes)
        nn.init.xavier_normal_(self.task5.weight) 

        
    def forward(self, x):
        """ Forward pass over final layer.

        Parameters
        ----------
        x
            tensor, output of the ResNet just before the final layer.
        Returns
        -------
        tensor_outputs
            tensor, list of class probabilities for each task
        """
        base = self.resnet_base(x)
        base = self.bn1(base)
        base = self.bn2(self.x2(base))

        task1_out = F.softmax(self.task1(base), dim=1)
        task2_out = F.softmax(self.task2(base), dim=1)
        task3_out = F.softmax(self.task3(base), dim=1)
        task4_out = F.softmax(self.task4(base), dim=1)
        task5_out = F.softmax(self.task5(base), dim=1)


        # Arrange the outputs into a tensor such that the size is
        # Batch size x Task number x Class number
        outputs = [task1_out, task2_out, task3_out, task4_out, task5_out]
        tensor_outputs = torch.stack(outputs).permute(1, 0, 2)
        return tensor_outputs

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
        model.fc = nn.Linear(model.fc.in_features, 256)
        model = MultiTaskFinalLayer(model, num_classes, num_tasks)
    else:
        raise ValueError("Not a valid task type.")
    model = model.to(device)

    params_to_update = [
        param for name, param in model.named_parameters() if param.requires_grad
    ]

    return model, params_to_update


def get_test_predictions(model, test_dataloader, encoder):
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
    
    acc = running_corrects.double() / len(test_dataloader.dataset)

    print('\nTest accuracy: {}'.format(acc))
    return predicted_chars, acc


class UniformLinearLoss(torch.nn.Module):
    """ Uniformly averaging of all task loss where the task loss
    is the Cross Entropy Loss.
    """
    def __init__(self, num_tasks):
        super(UniformLinearLoss, self).__init__()
        self.num_tasks = num_tasks
        self.criterions = [nn.CrossEntropyLoss(reduction='sum') for _ in range(num_tasks)]
        
    def forward(self, outputs, labels):
        outputs_np = np.array([task_output.detach().numpy() for task_output in outputs])
        task_losses = torch.tensor(0.)
        for task in range(self.num_tasks):
            task_output = outputs_np[:, task, :]
            task_label = labels[:, task]
            criterion = self.criterions[task]
            task_losses += criterion(torch.from_numpy(task_output), task_label)

        task_losses = Variable(task_losses, requires_grad = True)
        return task_losses



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
    #criterion = UniformLinearLoss(num_tasks)
    #criterion = nn.CrossEntropyLoss(reduction='sum')
    criterion = [
        nn.CrossEntropyLoss(reduction='sum'),
        nn.CrossEntropyLoss(reduction='sum'),
        nn.CrossEntropyLoss(reduction='sum'),
        nn.CrossEntropyLoss(reduction='sum'),
        nn.CrossEntropyLoss(reduction='sum'),
    ]

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

    #optimizer = optim.SGD(params_to_update, lr=learning_rate, momentum=momentum)
    optimizer = optim.Adam(params_to_update, lr=learning_rate)

    # train model
    model, val_acc_history = train_model(
        model, dataloaders, criterion, optimizer, num_epochs=num_epochs, device=device
    )

    # plot the gradients for sanity check
    plt.tight_layout()
    plt.show()

    # get predictions
    test_chars, test_acc = get_test_predictions(model, test_dataloader, encoder)

    pdb.set_trace()


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        "-c", "--config", help="filepath to config json", default="./config.json"
    )
    ARGS = PARSER.parse_args()
    CONFIGPATH = ARGS.config
    main(CONFIGPATH)