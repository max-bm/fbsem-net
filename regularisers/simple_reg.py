"""
Author: Maxwell Buckmire-Monro
maxbuckmiremonro@gmail.com
"""

import torch
import torch.nn as nn


class IdentityMapping(nn.Module):
    """
    Custom PyTorch identity mapping which returns input tensor unchanged.

    Parameters
    ----------
    in_channels: int
        Dummy variable for compatibility with FBSEMNet.

    Attributes
    ----------
    Identity: nn.Identity
        Identity layer.
    """
    def __init__(self, in_channels=1):
        super(IdentityMapping, self).__init__()
        self.in_channels = in_channels
        self.Identity = nn.Identity()

    def forward(self, img: torch.Tensor, mr=None):
        """
        Forward pass through identity layer.

        Parameters
        ----------
        img: torch.tensor
            Input image.

        mr: bool
            Toggle for whether mr image supplied to regulariser.

        Returns
        -------
        torch.Tensor
            Unchanged input image tensor.
        """
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
        """
        Forward pass through 'zero layer.

        Parameters
        ----------
        img: torch.tensor
            Input image.

        mr: bool
            Toggle for whether mr image supplied to regulariser.

        Returns
        -------
        torch.Tensor
            Zero-filled image tensor.
        """
        return img * 0.
