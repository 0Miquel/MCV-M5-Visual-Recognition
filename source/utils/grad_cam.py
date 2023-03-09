import torch
import numpy as np

def get_grad_cam(grad_cam, model, x: torch.Tensor) -> np.ndarray:
    """
    Computes the grad cam for the given input. Assumes a batch of data.
    Extracted from: https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
    :param x: input tensor (batch_size, channels, height, width)
    :return: grad cam (batch_size, height, width)
    """
    assert grad_cam, "Grad cam is not enabled"
    # pull the gradients out of the model
    gradients = model.get_activations_gradient()
    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    # get the activations of the last convolutional layer
    activations = model.get_activations(x).detach()
    # weight the channels by corresponding gradients
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()
    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(heatmap.cpu(), 0)
    # normalize the heatmap
    heatmap /= torch.max(heatmap)
    return heatmap
