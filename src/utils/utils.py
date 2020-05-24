import numpy as np
import torch
import matplotlib.pyplot as plt


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
    """
    Given the predictions from the model, find the class with the highest
    probability. (Works for multi-label and single predictions)

    Parameters
    ----------
    outputs
        torch.tensor, model outputs

    Returns
    -------
    flat_preds
        torch.tensor, predictions flattened to be suitable for loss function
    """
    if outputs.dim() == 2:
        _, flat_preds = torch.max(outputs, 1)
    elif outputs.dim() == 3:
        # if outputs dimension is 3, this was a multi-task learning
        flat_preds = torch.stack([torch.max(task_outputs, 1)[1] for task_outputs in outputs])
    else:
        raise ValueError("Dimension of the output must be 2 or 3.")

    return flat_preds

def get_loss(criterions, outputs, labels, alphas):
    """
    Trains the Deep Learning model. Keeps track of the best model according to the
    holdout validation set.

    Parameters
    ----------
    criterions
        list, list of loss function used in optimization of each task
    outputs
        torch.tensor, model outputs
    labels
        torch.tensor, true labels
    alphas
        list, weights in the linear combination of loss functions

    Returns
    -------
    loss
        torch.tensor, combined loss of all tasks
    """
    if np.sum(alphas) != 1:
        raise ValueError("Alpha values should sum up to 1.")
    if (alphas < 0).any():
        raise ValueError("All alpha values should be non-negative.")

    if outputs.dim() == 3:
        loss = torch.tensor(0.)
        for t, alpha in enumerate(alphas):
            loss_ind = criterions[t](outputs[:, t, :], labels[:, t])
            loss += alpha * loss_ind
    elif outputs.dim() == 2:
        loss = criterions[0](outputs, labels)
    
    return loss

