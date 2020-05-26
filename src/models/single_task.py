import torch
import torch.nn as nn
import torch.nn.functional as F


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
        output
            tensor, class probabilities
        """
        output = F.softmax(self.layer(x), dim=1)
        return output
