"""
Author: Maxwell Buckmire-Monro
maxwell.monro@kcl.ac.uk
"""

import torch
from pet_system_model import PETSystemModel
from differentiable_functions import ForwardModel, BackwardModel

def em_update(img: torch.Tensor, sino: torch.Tensor, sens_img: torch.Tensor,
    system_model: PETSystemModel):
    """
    Function which computes a single EM update given a current image estimate,
    the measured sinogram data, system sensitivity image (note, this is
    dependent on the system model, but is precomputed to save computation time),
    and the PET system model.
    """
    forward_model = ForwardModel.apply
    backward_model = BackwardModel.apply

    device = 'cpu' if theta.get_device() == -1 else theta.get_device()
    forwardprojection = forward_model(img, system_model)
    fp_mask = (forwardprojection != 0)
    ratio_sino = torch.zeros_like(sino).to(device).float()
    ratio_sino[fp_mask] = sino [fp_mask] / forwardprojection[fp_mask]
    backprojection = backward_model(ratio_sino, system_model)
    bp_mask = (backprojection != 0)
    updated_img = torch.zeros_like(img).to(device).float()
    updated_img[bp_mask] = backprojection[bp_mask] * img[bp_mask] / sens_img[bp_mask]
    return update
