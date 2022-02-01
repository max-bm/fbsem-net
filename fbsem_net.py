"""
Author: Maxwell Buckmire-Monro
maxwell.monro@kcl.ac.uk
"""

import torch
import torch.nn as nn
import numpy as np
from torch_pet.system_model.pet_system_model import PETSystemModel
from torch_pet.system_model.differentiable_functions import ForwardModel, BackwardModel
from torch_pet.regularisers.simple_reg import IdentityMapping, ZeroMapping
from torch_pet.utilities import DictDataset, nrmse, em_update
import matplotlib.pyplot as plt
import copy
import time
import random


class FBSEMNet(nn.Module):
    """
    Class to define FBSEM-Net architecture. Define forward function uniquely for
    training (i.e. target == None) and for testing (i.e. target != None) - this
    handles multiple noisy realisations during testing and calculates NRMSE.

    Parameteres
    -----------
    system_model: PETSystemModel
        PETSystemModel object which defines imaging system.

    regulariser: nn.Module
        PyTorch network defining regularisation block.

    n_mods: int
        Number of modules in the overall RNN.

    fixed_beta: float
        Fixed fusion coefficient. False by default, giving random and
        learnable fusion coefficient.

    to_convergence: bool
        Toggle to run model until output has converged, up to a maximum
        number of modules given by n_mod.

    Attributes
    ----------
    beta: torch.Parameter
        Learnable fusion weight.
    """
    def __init__(self, system_model: PETSystemModel, regulariser, n_mods: int,
        fixed_beta = False, to_convergence=False):
        super(FBSEMNet, self).__init__()
        self.system_model = system_model
        self.regulariser = regulariser
        self.n_mods = n_mods
        self.to_convergence = to_convergence
        self.register_parameter(name='beta', param=nn.Parameter(torch.rand(1),
            requires_grad=True))
        if isinstance(fixed_beta, float):
            self.beta.data = torch.Tensor([fixed_beta])

    def forward(self, sino: torch.Tensor, mr=None, target=None):
        """
        Forward pass through FBSEM-Net.

        Parameters
        ----------
        sino: torch.Tensor
            Input measured sinogram tensor.

        mr: torch.Tensor
            Input corresponding MR tensor for anatomically-guided recon.

        target: torch.Tensor
            Targets of reconstruction, for test-time performance measures.

        Returns
        -------
        img: torch.Tensor
            Reconstructed image tensor.

        recon_dict: dict
            Reconstruction data.
        """
        device = 'cpu' if sino.get_device() == -1 else sino.get_device()
        batch_size = sino.shape[0]
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
            img = torch.ones(batch_size, 1, img_size[-2],
                img_size[-1]).to(device).float()
            # FBSEM loop
            for i in range(self.n_mods):
                out_em = out_reg = torch.zeros_like(img)
                # Loop through mini-batch for EM update
                # Potential to vectorise this within em_update function
                # definition
                for b in range(batch_size):
                    # EM block
                    out_em[b, 0, :, :] = \
                        em_update(img[b, 0, :, :], sino[b, 0, :, :],
                            sens_img, self.system_model)
                # Reg block - PyTorch handles mini-batch in parallel (I think)
                out_reg = self.regulariser(img.view(1, 1, img_size[-2], img_size[-1]), mr)
                # Fusion block - parallelised in fusion function definition
                img = fbsem_fusion(out_em, out_reg, inv_sens_img, self.beta)
            return img

        # Testing
        else:
            if self.to_convergence:
                mse_tracker = list()
            recon_dict = dict()
            recon_dict['nrmse_wrt_ref'] = np.zeros((self.n_mods, 1))
            recon_dict['nrmse_wrt_gt'] = np.zeros((self.n_mods, 1))
            recon_dict['bias_wrt_ref'] = np.zeros((self.n_mods, 1))
            recon_dict['sd_wrt_ref'] = np.zeros((self.n_mods, 1))
            recon_dict['bias_wrt_gt'] = np.zeros((self.n_mods, 1))
            recon_dict['sd_wrt_gt'] = np.zeros((self.n_mods, 1))
            recon_dict['nrmse_wrt_ref'][:] = np.nan
            recon_dict['nrmse_wrt_gt'][:] = np.nan
            recon_dict['bias_wrt_ref'][:] = np.Inf
            recon_dict['sd_wrt_ref'][:] = np.Inf
            recon_dict['bias_wrt_gt'][:] = np.Inf
            recon_dict['sd_wrt_gt'][:] = np.Inf

            n_realisations = sino.shape[2]
            sens_img = self.system_model.backward_model(
                torch.ones_like(sino[0, 0, 0, :, :])).to(device).float()
            sens_mask = (sens_img != 0)
            inv_sens_img = torch.zeros_like(sens_img).to(device).float()
            inv_sens_img[sens_mask] = 1. / sens_img[sens_mask]

            # Initialise image estimate
            img_size = sens_img.shape
            img = torch.ones(batch_size, 1, n_realisations, img_size[-2],
                img_size[-1]).to(device).float()
            # FBSEM Loop
            for i in range(self.n_mods):
                out_em = torch.zeros_like(img)
                out_reg = torch.zeros_like(img)
                # Loop through mini-batch for EM update
                # Potential to vectorise this within em_update function
                # definition
                for b in range(batch_size):
                    for r in range(n_realisations):
                        # EM block
                        out_em[b, 0, r, :, :] = \
                            em_update(img[b, 0, r, :, :], sino[b, 0, r, :, :],
                                sens_img, self.system_model)
                        out_reg = self.regulariser(img[b, 0, r, :, :].view(1, 1, img_size[-2], img_size[-1]), mr)
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

                recon_dict['nrmse_wrt_ref'][i], \
                    recon_dict['bias_wrt_ref'][i], \
                        recon_dict['sd_wrt_ref'][i] = \
                            nrmse(img, target[0])
                if recon_dict['nrmse_wrt_ref'][i] == \
                    min(recon_dict['nrmse_wrt_ref'][:]):
                    recon_dict['best_nrmse_wrt_ref'] = \
                        recon_dict['nrmse_wrt_ref'][i]
                    recon_dict['best_mod_wrt_ref'] = i + 1
                    recon_dict['best_recon_wrt_ref'] = \
                        img.detach().cpu().numpy()
                recon_dict['nrmse_wrt_gt'][i], \
                    recon_dict['bias_wrt_gt'][i], \
                        recon_dict['sd_wrt_gt'][i] = \
                            nrmse(img, target[1])
                if recon_dict['nrmse_wrt_gt'][i] == \
                    min(recon_dict['nrmse_wrt_gt'][:]):
                    recon_dict['best_nrmse_wrt_gt'] = \
                        recon_dict['nrmse_wrt_gt'][i]
                    recon_dict['best_mod_wrt_gt'] = i + 1
                    recon_dict['best_recon_wrt_gt'] = \
                        img.detach().cpu().numpy()

            recon_dict['final_recon'] = img.detach().cpu().numpy()
            recon_dict['final_nrmse_wrt_ref'] = nrmse(img, target[0])
            recon_dict['final_nrmse_wrt_gt'] = nrmse(img, target[1])
            return img, recon_dict


def fbsem_fusion(out_em: torch.Tensor, out_reg: torch.Tensor,
    inv_sens_img: torch.Tensor, beta: torch.nn.Parameter):
    """
    Function to compute fusion of EM update and regulariser block.

    Parameters
    ----------
    out_em: torch.Tensor
        EM update output tensor.

    out_reg: torch.Tensor
        Regulariser block output tensor.

    inv_sens_img: torch.Tensor
        Reciprocal of system sensitivity image.

    beta: torch.Parameter
        Learnable fusion weight.
    """
    return 2 * out_em / (1 - beta**2 * inv_sens_img * out_reg + \
        torch.sqrt((1 - beta**2 * inv_sens_img * out_reg)**2 + \
            4 * beta**2 * inv_sens_img * out_em))


def train_fbsem(model, train_loader, val_loader, model_name='', save_dir='',
                epochs=50, lr=3e-4, mr_scale=1):
    """
    Function for training FBSEM-Net, with concurrent training and validation
    loops.

    Parameters
    ----------
    model: FBSEMNet
        Trainable FBSEM-Net architecture.

    train_loader, val_loader: torch.utils.data.DataLoader
        PyTorch dataloaders for training and validation sets.

    model_name: str
        Name of model for saving.

    save_dir: str
        Path to directory for saving.

    epochs: int
        Number of training and validation epochs.

    lr: float
        Learning rate for training network parameters and beta.

    mr_scale: float
        Scaling factor for MR input values.

    Returns
    -------
    FBSEMNet
        Trained model with best performance on validation set.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loss_func = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss = list()
    val_loss = list()
    beta_var = list()
    training_models = list()
    best_model = copy.deepcopy(model)

    t0 = time.time()
    for e in range(epochs):
        t1 = time.time()
        epoch_train_loss = list()
        epoch_val_loss = list()

        # Training loop
        model.train()
        for i, sample in enumerate(train_loader):
            optimiser.zero_grad()
            # We choose a random sinogram from all possible noisy realisations
            # as a form of data augmentation to artificially increase the size
            # of the training set. This should help train the network to be
            # robust to noise in the data.
            n_realisations = sample['noisy_sino'].shape[2]
            idx = random.randint(0, n_realisations - 1)
            sino = sample['noisy_sino'][:, :, idx, :, :].float().to(device)
            target = sample['HD_target'].float().to(device)
            if model.regulariser.in_channels == 1:
                mr = None
            else:
                mr = sample['mr'].float().to(device)
                mr = mr_scale * mr / mr.max*()

            output = model(sino, mr)
            loss = loss_func(output, target)
            epoch_train_loss.append(loss.item())
            loss.backward()
            optimiser.step()

        train_loss.append(np.mean(epoch_train_loss))
        print('Epoch {}/{}: Training loss = {}, Time/epoch = {}s.'.format(e+1,
            epochs, round(train_loss[-1], 6), round(time.time()-t1, 2)))
        print('beta = {}.'.format(model.beta.clone().detach().cpu().numpy()))
        beta_var.append(model.beta.clone().detach().cpu().numpy())

        # Save model to list of training models
        checkpt = dict()
        checkpt['state_dict'] = model.state_dict()
        checkpt['beta'] = model.beta.data
        checkpt['beta_var'] = beta_var
        training_models.append(checkpt)

        # Validation loop
        with torch.no_grad():
            model.eval()
            for i, sample in enumerate(val_loader):
                # We validate on a static set, so now we use the same noisy
                # realisation each time
                sino = sample['noisy_sino'][:, :, 0, :, :].float().to(device)
                target = sample['HD_target'].float().to(device)
                if model.regulariser.in_channels == 1:
                    mr = None
                else:
                    mr = sample['mr'].float().to(device)
                    mr = mr_scale * mr / mr.max*()

                output = model(sino, mr)
                loss = loss_func(output, target)
                epoch_val_loss.append(loss.item())

            val_loss.append(np.mean(epoch_val_loss))
            print('Validation loss = {}.'.format(round(val_loss[-1], 6)))

        if e > 1 and val_loss[-1] == min(val_loss):
            print('New best model: epoch {}'.format(e+1))
            best_model = copy.deepcopy(model)

    # Training finished
    print('Training time: {}min'.format(round((time.time() - t0) / 60, 2)))
    final_checkpt = dict()
    final_checkpt['best_model'] = best_model.state_dict()
    final_checkpt['train_loss'] = train_loss
    final_checkpt['val_loss'] = val_loss
    final_checkpt['beta_var'] = beta_var
    final_checkpt['training_models'] = training_models
    torch.save(final_checkpt, '{}{}_trained_{}-epochs.pth'.format(save_dir,
        model_name, epochs))
    return best_model


def test_fbsem(model, test_loader, model_name='', save_dir='', mr_scale=1):
    """
    Function for testing FBSEM-Net.

    Parameters
    ----------
    model: FBSEMNet
        Trainable FBSEM-Net architecture.

    test_loader: torch.utils.data.DataLoader
        PyTorch dataloader for test set.

    model_name: str
        Name of model for saving.

    save_dir: str
        Path to directory for saving.

    mr_scale: float
        Scaling factor for MR input values.

    Returns
    -------
    dict
        Dictionary of test peformance results.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    with torch.no_grad():
        model.eval()
        for i, sample in enumerate(test_loader):
            # We validate on a static set, so now we use the same noisy
            # realisation each time
            sino = sample['noisy_sino'].float().to(device)
            target = sample['HD_target'].float().to(device)
            ground_truth = sample['pet_gt'].float().to(device)
            if model.regulariser.in_channels == 1:
                mr = None
            else:
                mr = sample['mr'].float().to(device)
                mr = mr_scale * mr / mr.max*()

            print(sino.shape)
            output, test_results = model(sino, mr, target=(target,
                ground_truth))
            test_results['noisy_sino'] = sino
            test_results['HD_target'] = target.detach().cpu().numpy()
            test_results['pet_gt'] = ground_truth.detach().cpu().numpy()
            test_results['HD_counts'] = sample['HD_counts']
            test_results['LD_counts'] = sample['LD_counts']
            test_results['mlem_iters'] = sample['mlem_iters']
            test_results['factor'] = sample['factor']
            test_results['final_recon'] = output.detach().cpu().numpy()
            test_results['final_nrmse_wrt_ref'] = nrmse(output, target)
            test_results['final_nrmse_wrt_gt'] = nrmse(output, ground_truth)

    torch.save(test_results, save_dir + model_name + '_test_results.pth')
    return test_results
