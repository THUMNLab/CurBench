import numpy as np
from scipy.special import lambertw
import torch

from .base import BaseTrainer, BaseCL



class SuperLoss(BaseCL):
    """SuperLoss.
    
    SuperLoss: A Generic Loss for Robust Curriculum Learning
    https://proceedings.neurips.cc/paper/2020/file/2cfa8f9e50e0f510ede9d12338a5f564-Paper.pdf
    """
    def __init__(self, tau, lam, fac):
        super(SuperLoss, self).__init__()

        self.name = 'superloss'
        self.tau = tau
        self.lam = lam
        self.fac = fac


    def loss_curriculum(self, outputs, labels, indices, **kwargs):
        loss = self.criterion(outputs, labels)
        origin_loss = loss.detach().cpu().numpy()

        if self.tau == 0.0: self.tau = origin_loss.mean()
        if self.fac > 0.0: self.tau = self.fac * self.tau + (1.0 - self.fac) * origin_loss.mean()

        beta = (origin_loss - self.tau) / self.lam
        gamma = -2.0 / np.exp(1.0) + 1e-12
        sigma = np.exp(-lambertw(0.5 * np.maximum(beta, gamma)).real)
        sigma = torch.from_numpy(sigma).to(self.device)
        super_loss = (loss - self.tau) * sigma + self.lam * (torch.log(sigma) ** 2)
        return torch.mean(super_loss)



class SuperLossTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, gpu_index, num_epochs, random_seed, 
                 tau, lam, fac):
        
        cl = SuperLoss(tau, lam, fac)
        
        super(SuperLossTrainer, self).__init__(
            data_name, net_name, gpu_index, num_epochs, random_seed, cl)