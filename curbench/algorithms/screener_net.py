import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.data.batch import Batch as pygBatch

from .base import BaseTrainer, BaseCL



class ScreenerNet(BaseCL):
    """ScreenerNet CL Algorithm. 
    
    Screenernet: Learning self-paced curriculum for deep neural networks. https://arxiv.org/pdf/1801.00904
    """
    def __init__(self, M):
        super(ScreenerNet, self).__init__()

        self.name = 'screener_net'
        self.M = M
        

    def data_prepare(self, loader, **kwargs):
        super().data_prepare(loader)


    def model_prepare(self, net, device, epochs, criterion, optimizer, lr_scheduler, **kwargs):
        super().model_prepare(net, device, epochs, criterion, optimizer, lr_scheduler)
        
        self.snet = copy.deepcopy(self.net)
        self.fc = nn.Sequential(nn.Linear(self.net.num_classes, 1), nn.Sigmoid()).to(self.device)
        self.optimizer_s = copy.deepcopy(optimizer)
        self.optimizer_s.add_param_group({'params': self.snet.parameters()})
        self.optimizer_s.add_param_group({'params': self.fc.parameters()})
        self.optimizer_s.param_groups.pop(0)


    def loss_curriculum(self, outputs, labels, indices, **kwargs):
        if not isinstance(indices, list): indices = indices.tolist()
        loss = self.criterion(outputs, labels)

        self.snet.train()
        self.fc.train()
        data = next(iter(self._dataloader(Subset(self.dataset, indices), shuffle=False)))
        if isinstance(data, list):          # data from torch.utils.data.Dataset
            inputs = data[0].to(self.device)
            weights = self.fc(self.snet(inputs)).squeeze()
        elif isinstance(data, dict):        # data from datasets.arrow_dataset.Dataset
            inputs = {k: v.to(self.device) for k, v in data.items() 
                      if k not in ['labels', 'indices']}
            weights = self.fc(self.snet(**inputs)[0]).squeeze()
        elif isinstance(data, pygBatch):    # data from torch_geometric.datasets
            inputs = data.to(self.device)
            weights = self.fc(self.snet(inputs)).squeeze()
        else:
            not NotImplementedError()

        self.optimizer_s.zero_grad()
        loss_s = ((1.0 - weights) ** 2.0) * loss.detach() + (weights ** 2.0) * F.relu(self.M - loss.detach())
        torch.mean(loss_s).backward()
        self.optimizer_s.step()
        return torch.mean(loss * weights.detach())


class ScreenerNetTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, gpu_index, num_epochs, random_seed, M):
        
        cl = ScreenerNet(M)

        super(ScreenerNetTrainer, self).__init__(
            data_name, net_name, gpu_index, num_epochs, random_seed, cl)