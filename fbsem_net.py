"""
Author: Maxwell Buckmire-Monro
maxwell.monro@kcl.ac.uk
"""

import torch
import torch.nn
from pet_system_model import PETSystemModel
from differentiable_functions import ForwardModel, BackwardModel

def fbsem_fusion(out_em: torch.Tensor, out_reg: torch.Tensor,
    inv_sens_img: torch.Tensor, beta: torch.Parameter):
    """
    Function to compute fusion of output of EM update and of regulariser block.
    """
    return 2 * out_em / (1 - beta**2 * inv_sens_img * out_reg +
        torch.sqrt((1 - beta**2 * inv_sens_img * out_reg)**2 +
            4 * beta**2 * inv_sens_img * out_em))
