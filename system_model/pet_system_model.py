"""
Author: Maxwell Buckmire-Monro
maxbuckmiremonro@gmail.com
"""

import torch
import torch.nn as nn
from skimage.transform import radon, iradon

class PETSystemModel(nn.Module):
    """
    Class which defines the PET system model and implements the forward and
    backward models.

    Note: we employ here the scikit-image implementation of the radon (and its
    inverse, iradon) transform, though there are other options, as well as the
    possibility of implementing it from scratch in PyTorch.

    Parameters
    ----------
    projection_angles: list
        List of angles through which radon transform is applied.
    """
    def __init__(self, projection_angles: list):
        super(PETSystemModel, self).__init__()
        self.projection_angles = projection_angles

    def forward_model(self, img: torch.Tensor):
        """
        Forward model of input object to sinogram, given system model.

        Parameters
        ----------
        img: torch.Tensor
            Input image to be projected into sinogram space.

        Returns
        -------
        torch.tensor
            Corresponding sinogram.
        """
        return torch.from_numpy(radon(img.detach().cpu().numpy(),
            self.projection_angles)).float()

    def backward_model(self, sino: torch.Tensor):
        """
        Backward model of input sino to image, given system model.

        Parameters
        ----------
        sino: torch.Tensor
            Input sinogram to be projected into image space.

        Returns
        -------
        torch.tensor
            Corresponding image.
        """
        return torch.from_numpy(iradon(sino.detach().cpu().numpy(),
            filter_name=None)).float()
