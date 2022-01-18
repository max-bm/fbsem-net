"""
Author: Maxwell Buckmire-Monro
maxwell.monro@kcl.ac.uk
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    """
    Splits input image into patches (tokens) and embeds them into 'arbitrary' dimension.

    Parameters
    ----------
    img_size: int
        Size of the (square) image.
    
    patch_size: int
        Size of the (square) patches.

    in_channels: int
        Number of input channels. Greyscale = 1.

    embedding_dim: int
        Size of arbitrary embedding dimension.

    Attributes
    ----------
    n_patches: int
        Number of patches.

    embedding_proj: nn.Conv2d
        Convolutional layer that performs splitting and embedding.
    """
    def __init__(self, img_size=144, patch_size=4, in_channels=1, embedding_dim=96):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.embedding_proj = nn.Conv2d(
            in_channels, 
            embedding_dim, 
            kernel_size=patch_size, 
            stride=patch_size)

    def forward(self, img):
        """
        Forward pass.

        Parameters
        ----------
        img: torch.Tensor
            Input image to be split into embedded patches.
            Shape: (batch_size, in_channels, img_size, img_size)

        Returns
        -------
        torch.Tensor
            Embedded patches.
            Shape: (batch_size, n_patches, embedding_dim)
        """
        out = self.embedding_proj(img) # (batch_size, embedding_dim, n_patches ** 0.5, n_patches ** 0.5)
        out = out.flatten(2) # (batch_size, embedding_dim, n_patches)
        out = out.transpose(1, 2) # (batch_size, n_patches, embedding_dim)

        return out
