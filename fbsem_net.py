"""
Author: Maxwell Buckmire-Monro
maxwell.monro@kcl.ac.uk
"""

import torch
import torch.nn as nn
import numpy as np
from pet_system_model import PETSystemModel
from differentiable_functions import ForwardModel, BackwardModel
from torch_em import em_update
from utilities import DictDataset, nrmse
import matplotlib.pyplot as plt

def fbsem_fusion(out_em: torch.Tensor, out_reg: torch.Tensor,
    inv_sens_img: torch.Tensor, beta: torch.nn.Parameter):
    """
    Function to compute fusion of output of EM update and of regulariser block.
    """
    return 2 * out_em / (1 - beta**2 * inv_sens_img * out_reg + \
        torch.sqrt((1 - beta**2 * inv_sens_img * out_reg)**2 + \
            4 * beta**2 * inv_sens_img * out_em))


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


class FBSEMNet(nn.Module):
    """

    """
    def __init__(self, system_model: PETSystemModel, regulariser, n_mods: int,
        batch_size: int, fixed_beta: float, to_convergence=False):
        super(FBSEMNet, self).__init__()
        self.system_model = system_model
        self.regulariser = regulariser
        self.n_mods = n_mods
        self.batch_size = batch_size
        self.to_convergence = to_convergence
        self.register_parameter(name='beta', param=nn.Parameter(torch.rand(1),
            requires_gard=True))
        if isinstance(fixed_beta, float):
            self.beta.data = torch.Tensor([fixed_beta])

    def forward(self, sino: torch.Tensor, mr=None, device='cpu', target=None):
        # Training/validation
        if target is None:
            # Generate sensitivity image and its reciprocal
            sens_img = self.system_model.backward_model(
                torch.ones_like(sino[0, 0, :, :])).to(device).float()
            sens_mask = (sens_img != 0)
            inv_sens_img = torch.zeros_like(sens_img).to(device).float()
            inv_sens_img[sens_mask] = 1. / sens_img[sens_mask]

            # Initialise image estimate
            img_size = sens_img.shape
            img = torch.ones(self.batch_size, 1, img_size[-2],
                img_size[-1]).to(device).float()
            # FBSEM loop
            for i in range(self.n_mods):
                out_em = out_reg = torch.zeros_like(img)
                # Loop through mini-batch for EM update
                # Potential to vectorise this within em_update function
                # definition
                for b in range(self.batch_size):
                    # EM block
                    out_em[b, 0, :, :] = \
                        em_update(img[b, 0, :, :], sino[b, 0, :, :],
                            sens_img, self.system_model)
                # Reg block - PyTorch handles mini-batch in parallel (I think)
                out_reg = self.regulariser(img, mr)
                # Fusion block - parallelised in fusion function definition
                img = fbsem_fusion(out_em, out_reg, inv_sens_img, self.beta)
            return img

        # Testing
        else:
            if self.to_convergence:
                mse_tracker = list()
            recon_dict = dict()
            recon_dict['test_nrmse'] = \
                recon_dict['bias-variance_wrt_ref'] = \
                    recon_dict['bias-variance_wrt_gt'] = \
                        np.array((np.zeros((self.n_mods, 1)),
                            np.zeros((self.n_mods, 1))))
            recon_dict['test_nrmse'][:] = np.nan
            recon_dict['best_nrmse_wrt_ref'] = \
                recon_dict['best_nrmse_wrt_gt'] = \
                    np.Inf

            n_realisations = sino.shape[2]
            sens_img = self.system_model.backward_model(
                torch.ones_like(sino[0, 0, 0, :, :])).to(device).float()
            sens_mask = (sens_img != 0)
            inv_sens_img = torch.zeros_like(sens_img).to(device).float()
            inv_sens_img[sens_mask] = 1. / sens_img[sens_mask]

            # Initialise image estimate
            img_size = sens_img.shape
            img = torch.ones(self.batch_size, 1, n_realisations, img_size[-2],
                img_size[-1]).to(device).float()
                # FBSEM Loop
            for i in range(self.n_mods):
                out_em = out_reg = torch.zeros_like(img)
                # Loop through mini-batch for EM update
                # Potential to vectorise this within em_update function
                # definition
                for b in range(self.batch_size):
                    for r in range(n_realisations):
                        # EM block
                        out_em[b, 0, r, :, :] = \
                            em_update(img[b, 0, r, :, :], sino[b, 0, r, :, :],
                                sens_img, self.system_model)
                        out_reg = self.regulariser(img[b, 0, r, :, :], mr)
                # Fusion block - parallelised in fusion function definition
                temp_img = fbsem_fusion(out_em, out_reg,
                    inv_sens_img, self.beta)
                if self.to_convergence:
                    rel_error = np.mean(np.abs(temp_img.detach().cpu().numpy() -
                        img.detach().cpu().numpy())) / np.mean(
                            img.detach().cpu().numpy())
                    mse_tracker.append(rel_error)
                    if i > 10 and np.mean(np.array(mse_tracker[-10:])) < 1e-4:
                        recon_dict['n_mods'] = i
                        img = temp_img
                        break
                img = temp_img

                recon_dict['test_nrmse'][0, i], \
                    recon_dict['bias-variance_wrt_ref'][0, i], \
                        recon_dict['bias-variance_wrt_ref'][1, i] = \
                            nrmse(img, target[0])
                if recon_dict['test_nrmse'][0, i] == \
                    min(recon_dict['test_nrmse'][0, :]):
                    recon_dict['best_nrmse_wrt_ref'] = \
                        recon_dict['test_nrmse'][0, i]
                    recon_dict['best_mod_wrt_ref'] = i + 1
                    recon_dict['best_recon_wrt_ref'] = \
                        img.detach().cpu().numpy()
                recon_dict['test_nrmse'][1, i], \
                    recon_dict['bias-variance_wrt_gt'][0, i], \
                        recon_dict['bias-variance_wrt_gt'][1, i] = \
                            nrmse(img, target[1])
                if recon_dict['test_nrmse'][1, i] == \
                    min(recon_dict['test_nrmse'][1, :]):
                    recon_dict['best_nrmse_wrt_gt'] = \
                        recon_dict['test_nrmse'][1, i]
                    recon_dict['best_mod_wrt_gt'] = i + 1
                    recon_dict['best_recon_wrt_gt'] = \
                        img.detach().cpu().numpy()
            return img, recon_dict
