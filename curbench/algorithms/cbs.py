import math
import torch
import torch.nn as nn

from .base import BaseTrainer, BaseCL
from .utils import KernelConv2d



class CBS(BaseCL):
    """Curriculum by Smoothing. 
    
    Curriculum by Smoothing. 
    https://proceedings.neurips.cc/paper/2020/file/f6a673f09493afcd8b129a0bcf1cd5bc-Paper.pdf
    https://www.github.com/pairlab/CBS
    """
    def __init__(self, kernel_size, start_std, grow_factor, grow_interval):
        super(CBS, self).__init__()

        self.name = 'cbs'
        self.epoch = 0

        self.kernel_size = kernel_size
        self.std = start_std
        self.grow_factor = grow_factor
        self.grow_interval = grow_interval


    def model_prepare(self, net, device, epochs, criterion, optimizer, lr_scheduler, **kwargs):
        super().model_prepare(net, device, epochs, criterion, optimizer, lr_scheduler)
        self.replace_conv2d(self.net)
    

    def replace_conv2d(self, model):
        for name, module in model.named_children():
            if isinstance(module, nn.Conv2d):
                setattr(model, name, KernelConv2d(module, self.kernel_size, self.std).to(self.device))
            else:
                self.replace_conv2d(module)


    def model_curriculum(self, **kwargs):
        self.epoch += 1

        if self.epoch > 1 and (self.epoch - 1) % self.grow_interval == 0:
            self.std *= self.grow_factor

        for _, module in self.net.named_modules():
            if isinstance(module, KernelConv2d):
                module.model_curriculum(self.std)
        return self.net.to(self.device)



class CBSTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, gpu_index, num_epochs, random_seed,
                 kernel_size, start_std, grow_factor, grow_interval):

        cl = CBS(kernel_size, start_std, grow_factor, grow_interval)

        super(CBSTrainer, self).__init__(
            data_name, net_name, gpu_index, num_epochs, random_seed, cl)
