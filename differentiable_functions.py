"""
Author: Maxwell Buckmire-Monro
maxwell.monro@kcl.ac.uk
"""

import torch
from pet_system_model import PETSystemModel

class ForwardModel(torch.autograd.Function):
    """
    A class which defines a differentiable forward model for a given PET system
    model.
    """
    @staticmethod
    def forward(ctx, img: torch.Tensor, system_model: PETSystemModel):
        ctx.device = 'cpu' if img.get_device == -1 else img.get_device
        ctx.system_model = system_model
        return system_model.forward_model(img).to(ctx.device).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return ctx.system_model.backward_model(
            grad_output).to(ctx.device).float()


class BackwardModel(torch.autograd.Function):
    """
    A class which defines a differentiable backward model for a given PET system
    model.
    """
    @staticmethod
    def forward(ctx, sino: torch.Tensor, system_model: PETSystemModel):
        ctx.device = 'cpu' if sino.get_device == -1 else sino.get_device
        ctx.system_model = system_model
        return system_model.backward_model(sino).to(ctx.device).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return ctx.system_model.forward_model(
            grad_output).to(ctx.device).float()
