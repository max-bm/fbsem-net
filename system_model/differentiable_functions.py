"""
Author: Maxwell Buckmire-Monro
maxwell.monro@kcl.ac.uk
"""

import torch
from torch_pet.system_model.pet_system_model import PETSystemModel

class ForwardModel(torch.autograd.Function):
    """
    A class which defines a differentiable forward model for a given PET system
    model.
    """
    @staticmethod
    def forward(ctx, img: torch.Tensor, system_model: PETSystemModel):
        """
        Forward pass through ForwardModel layer.

        Parameters
        ----------
        img: torch.Tensor
            Input image tensor for forward modelling.

        system_model: PETSystemModel
            PETSystemModel object which defines imaging system.

        Returns
        -------
        torch.Tensor
            Corresponding sinogram tensor.
        """
        ctx.device = 'cpu' if img.get_device() == -1 else img.get_device()
        ctx.system_model = system_model
        return system_model.forward_model(img).to(ctx.device).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return ctx.system_model.backward_model(
            grad_output).to(ctx.device).float(), None


class BackwardModel(torch.autograd.Function):
    """
    A class which defines a differentiable backward model for a given PET system
    model.
    """
    @staticmethod
    def forward(ctx, sino: torch.Tensor, system_model: PETSystemModel):
        """
        Forward pass through BackwardModel layer.

        Parameters
        ----------
        sino: torch.Tensor
            Input sino tensor for backward modelling.

        system_model: PETSystemModel
            PETSystemModel object which defines imaging system.

        Returns
        -------
        torch.Tensor
            Corresponding image tensor.
        """
        ctx.device = 'cpu' if sino.get_device() == -1 else sino.get_device()
        ctx.system_model = system_model
        return system_model.backward_model(sino).to(ctx.device).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return ctx.system_model.forward_model(
            grad_output).to(ctx.device).float(), None
