"""
Author: Maxwell Buckmire-Monro
maxwell.monro@kcl.ac.uk
"""

import torch
import torch.nn as nn
from skimage.transform import radon, iradon

class PETSystemModel(nn.Module):
    """
    Class which defines the PET system model and implements the forward and
    backward models. Defining the system model as its own class allows greater
    modularity of future code - it will be entirely sufficient to make any
    changes to the system model (e.g. inclusion of attenuation correction)
    within this class only.

    Note: we employ here the scikit-image implementation of the radon (and its
    inverse, iradon) transform, though there are other options, as well as the
    possibility of implementing it from scratch in PyTorch.
    """
    def __init__(self, projection_angles: list):
        super(PETSystemModel, self).__init__()
        self.projection_angles = projection_angles

    def forward_model(self, img: torch.Tensor):
        return torch.from_numpy(radon(img.detach().cpu().numpy(),
            self.projection_angles)).float()

    def backward_model(self, sino: torch.Tensor):
        return torch.from_numpy(iradon(sino.detach().cpu().numpy(),
            filter_name=None)).float()
