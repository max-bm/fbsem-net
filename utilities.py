"""
Author: Maxwell Buckmire-Monro
maxwell.monro@kcl.ac.uk
"""

import torch
import torch.nn as nn
import numpy as np

def nrmse(recons: torch.Tensor, target: torch.Tensor):
    """
    Function to compute NRMSE of reconstructions in comparison to given target.
    """
    recons = recons.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    mean_img = np.mean(recons, axis=2)
    repeat_mean_img = np.repeat(mean_img[:, :, np.newaxis, :, :],
        recons.shape[2], axis=2)
    bias = np.sqrt(np.sum(np.square(mean_img - target)) / 
        np.sum(np.square(target)))
    std_dev = np.sqrt(np.mean(np.sum(np.square(repeat_mean_img - recons),
        axis=(0, 1, 3, 4))) / np.sum(np.square(target)))
    return np.sqrt(bias**2 + std_dev**2), bias, std_dev


class DictDataset(torch.utils.data.Dataset):
    """
    Custom PyTorch dataset for handling list of dictionaries for dataloader.
    """
    def __init__(self, dict_list: list):
        self.dict_list = dict_list

    def __getitem__(self, idx):
        return self.dict_list[idx]

    def __len__(self):
        return len(self.dict_list)


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