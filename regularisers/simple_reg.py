"""
Author: Maxwell Buckmire-Monro
maxwell.monro@kcl.ac.uk
"""

import torch
import torch.nn as nn


class IdentityMapping(nn.Module):
    """
    Custom PyTorch identity mapping which has internal dummy variable
    in_channels (for compatibility with other code).
    """
    def __init__(self, in_channels=1):
        super(IdentityMapping, self).__init__()
        self.in_channels = in_channels
        self.Identity = nn.Identity()

    def forward(self, img: torch.Tensor, mr=None):
        return self.Identity(img)


class ZeroMapping(nn.Module):
    """
    Custom PyTorch zero mapping which has internal dummy variable
    in_channels (for compatibility with other code).
    """
    def __init__(self, in_channels=1):
        super(ZeroMapping, self).__init__()
        self.in_channels = in_channels

    def forward(self, img: torch.Tensor, mr=None):
        return img * 0.
