import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.num_tasks = num_tasks
        self.resnet_base = resnet_base

        # resnet.fc was replaced with Linear (512, 256)
        # so self.bn1 is for that linear layer
        self.bn1 = nn.BatchNorm1d(256, eps = 2e-1)
        self.x2 =  nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x2.weight)
        self.bn2 = nn.BatchNorm1d(256, eps = 2e-1)
        
        self.sequence = nn.Sequential()
        for i in range(num_tasks):
            task_name = "task_" + str(i)
            task_layer = nn.Linear(256, num_classes)
            nn.init.xavier_normal_(task_layer.weight)
            self.sequence.add_module(task_name, task_layer)

        
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

        outputs = []
        for i in range(self.num_tasks):
            outputs.append(F.softmax((self.sequence[i](base)), dim=1))

        # Arrange the outputs into a tensor such that the size is
        # Batch size x Task number x Class number
        tensor_outputs = torch.stack(outputs).permute(1, 0, 2)

        return tensor_outputs

