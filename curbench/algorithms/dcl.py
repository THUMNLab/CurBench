import numpy as np
import torch

from .base import BaseTrainer, BaseCL
from .utils import SparseSGD



class DCL(BaseCL):
    """Data Parameters Curriculum Learning. 
    
    Data Parameters: A New Family of Parameters for Learning a Differentiable Curriculum. 
    https://proceedings.neurips.cc/paper/2019/file/926ffc0ca56636b9e73c565cf994ea5a-Paper.pdf
    https://github.com/apple/ml-data-parameters
    """
    def __init__(self, init_class_param, lr_class_param, wd_class_param, 
                 init_data_param, lr_data_param, wd_data_param):
        super(DCL, self).__init__()

        self.name = 'dcl'

        self.init_class_param = init_class_param
        self.lr_class_param = lr_class_param
        self.wd_class_param = wd_class_param
        self.init_data_param = init_data_param
        self.lr_data_param = lr_data_param
        self.wd_data_param = wd_data_param


    def model_prepare(self, net, device, epochs, criterion, optimizer, lr_scheduler, **kwargs):
        super().model_prepare(net, device, epochs, criterion, optimizer, lr_scheduler)
        self.class_size = self.net.num_classes
        
        if self.init_data_param:
            self.data_weights = torch.tensor(
                np.ones(self.data_size) * np.log(self.init_data_param),
                dtype=torch.float32, requires_grad=True, device=self.device
            )
            self.data_optimizer = SparseSGD([self.data_weights], 
                lr=self.lr_data_param, momentum=0.9, skip_update_zero_grad=True
            ) # add weight decay at loss instead of here
            self.data_optimizer.zero_grad()

        if self.init_class_param:
            self.class_weights = torch.tensor(
                np.ones(self.class_size) * np.log(self.init_class_param),
                dtype=torch.float32, requires_grad=True, device=self.device
            )
            self.class_optimizer = SparseSGD([self.class_weights], 
                lr=self.lr_class_param, momentum=0.9, skip_update_zero_grad=True
            ) # add weight decay at loss instead of here
            self.class_optimizer.zero_grad()


    def loss_curriculum(self, outputs, labels, indices, **kwargs):
        if self.init_data_param:
            # update last batch
            self.data_optimizer.step()
            self.data_weights.data.clamp_(min=np.log(1/20), max=np.log(20))
            # calculate current batch
            self.data_optimizer.zero_grad()
            data_weights = self.data_weights[indices]

        if self.init_class_param:
            # update last batch
            self.class_optimizer.step()
            self.class_weights.data.clamp_(min=np.log(1/20), max=np.log(20))
            # calculate current batch
            self.class_optimizer.zero_grad()
            class_weights = self.class_weights[labels]

        if self.init_data_param and self.init_class_param:
            sigma = torch.exp(data_weights) + torch.exp(class_weights)
            loss = self.criterion(outputs / sigma.view(-1, 1), labels)      \
                + (0.5 * self.wd_data_param * data_weights ** 2).sum()      \
                + (0.5 * self.wd_class_param * class_weights ** 2).sum()
        elif self.init_data_param:
            sigma = torch.exp(data_weights)
            loss = self.criterion(outputs / sigma.view(-1, 1), labels)      \
                + (0.5 * self.wd_data_param * data_weights ** 2).sum()
        elif self.init_class_param:
            sigma = torch.exp(class_weights)
            loss = self.criterion(outputs / sigma.view(-1, 1), labels)      \
                + (0.5 * self.wd_class_param * class_weights ** 2).sum()
        else:
            loss = self.criterion(outputs, labels)
        return torch.mean(loss)


class DCLTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, gpu_index, num_epochs, random_seed,
                 init_class_param, lr_class_param, wd_class_param, 
                 init_data_param, lr_data_param, wd_data_param):
        
        cl = DCL(init_class_param, lr_class_param, wd_class_param, init_data_param, lr_data_param, wd_data_param)
        
        super(DCLTrainer, self).__init__(
            data_name, net_name, gpu_index, num_epochs, random_seed, cl)




