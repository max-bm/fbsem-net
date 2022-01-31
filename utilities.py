"""
Author: Maxwell Buckmire-Monro
maxwell.monro@kcl.ac.uk
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


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


def plot_test_results(test_results):
    """
    Function for plotting test results from dictionary.
    """
    ld_counts = test_results['LD_counts'][0].numpy()[0]
    best_nrmse_wrt_ref = np.round(test_results['best_nrmse_wrt_ref'][0] * 100, 3)
    best_nrmse_wrt_gt = np.round(test_results['best_nrmse_wrt_gt'][0] * 100, 3)
    final_nrmse_wrt_ref = np.round(test_results['final_nrmse_wrt_ref'][0] * 100, 3)
    final_nrmse_wrt_gt = np.round(test_results['final_nrmse_wrt_gt'][0] * 100, 3)
    final_recon_img = test_results['final_recon'][0, 0, 0, 20:-20, 20:-20]
    ground_truth = test_results['pet_gt'][0, 0, 20:-20, 20:-20]
    nrmse_wrt_ref = test_results['nrmse_wrt_ref']
    nrmse_wrt_gt = test_results['nrmse_wrt_gt']
    n_mods = len(nrmse_wrt_ref)

    fig, ax = plt.subplots(figsize=(16, 10), nrows=1, ncols=2)
    ax[0].imshow(final_recon_img, vmax=ground_truth.max(), cmap='Greys')
    ax[0].set_title(
        'Final reconstruction\nNRMSE wrt HQ Ref = {}\nNRMSE wrt GT = {}\nCounts = {}k'.format(
            final_nrmse_wrt_ref, final_nrmse_wrt_gt, int(round(ld_counts / 1e3))))
    ax[1].plot(np.arange(n_mods) + 1, nrmse_wrt_ref, label='HQ Reference')
    ax[1].plot(np.arange(n_mods) + 1, nrmse_wrt_gt, label='Ground Truth')
    ax[1].set_xlim(left=1)
    ax[1].set_ylim(bottom=0)
    ax[1].legend(title='wrt')
    ax[1].set_title(
        'NRMSE vs. module\nBest NRMSE wrt HQ Ref = {}\nBest NRMSE wrt GT = {}'.format(
            best_nrmse_wrt_ref, best_nrmse_wrt_gt))
    plt.show()
