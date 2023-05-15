import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset

from .base import BaseTrainer, BaseCL
from .utils import set_parameter



class DDS(BaseCL):
    """
    
    Optimizing data usage via differentiable rewards. http://proceedings.mlr.press/v119/wang20p/wang20p.pdf
    """
    def __init__(self, eps):
        super(DDS, self).__init__()

        self.name = 'dds'
        self.eps = eps


    def _meta_split(self):
        split = self.data_size // 10
        indices = list(range(self.data_size))
        np.random.shuffle(indices)
        train_idx, meta_idx = indices[split:], indices[:split]
        self.train_dataset = Subset(self.dataset, train_idx)
        self.meta_dataset = Subset(self.dataset, meta_idx)


    def data_prepare(self, loader, **kwargs):
        super().data_prepare(loader)
        self._meta_split()
        self.meta_loader = self._dataloader(self.meta_dataset)
        self.meta_iter = iter(self.meta_loader)


    def model_prepare(self, net, device, epochs, criterion, optimizer, lr_scheduler, **kwargs):
        super().model_prepare(net, device, epochs, criterion, optimizer, lr_scheduler)
        
        self.scorer = copy.deepcopy(self.net)
        self.fc = nn.Sequential(nn.Linear(self.net.num_classes, 1), nn.Sigmoid()).to(self.device)
        self.optimizer_s = copy.deepcopy(optimizer)
        self.optimizer_s.add_param_group({'params': self.scorer.parameters()})
        self.optimizer_s.add_param_group({'params': self.fc.parameters()})
        self.optimizer_s.param_groups.pop(0)


    def data_curriculum(self, **kwargs):
        return self._dataloader(self.train_dataset, shuffle=True)


    def loss_curriculum(self, outputs, labels, indices, **kwargs):
        if not isinstance(indices, list): indices = indices.tolist()
        loss = self.criterion(outputs, labels)

        meta_net = copy.deepcopy(self.net)
        meta_net.train()
        meta_net.zero_grad()
        try:
            meta_data = next(self.meta_iter)
        except StopIteration:
            self.meta_iter = iter(self.meta_loader)
            meta_data = next(self.meta_iter)
        if isinstance(meta_data, list):          # data from torch.utils.data.Dataset
            meta_inputs = meta_data[0].to(self.device)
            meta_labels = meta_data[1].to(self.device)
            meta_outputs = meta_net(meta_inputs)
        elif isinstance(meta_data, dict):        # data from datasets.arrow_dataset.Dataset
            meta_inputs = {k: v.to(self.device) for k, v in meta_data.items() 
                           if k not in ['labels', 'indices']}
            meta_labels = meta_data['labels'].to(self.device)
            meta_outputs = meta_net(**meta_inputs)[0]
        elif isinstance(meta_data, pygBatch):    # data from torch_geometric.datasets
            meta_inputs = meta_data.to(self.device)
            meta_labels = meta_data.y.to(self.device)
            meta_outputs = meta_net(meta_inputs)
        else:
            not NotImplementedError()
        meta_loss = self.criterion(meta_outputs, meta_labels)
        meta_grads = torch.autograd.grad(torch.mean(meta_loss), (meta_net.parameters()))

        train_data = next(iter(self._dataloader(Subset(self.dataset, indices), shuffle=False)))
        if isinstance(train_data, list):          # data from torch.utils.data.Dataset
            train_inputs = train_data[0].to(self.device)
            train_labels = train_data[1].to(self.device)
            train_outputs = meta_net(train_inputs)
        elif isinstance(train_data, dict):        # data from datasets.arrow_dataset.Dataset
            train_inputs = {k: v.to(self.device) for k, v in train_data.items() 
                           if k not in ['labels', 'indices']}
            train_labels = train_data['labels'].to(self.device)
            train_outputs = meta_net(**train_inputs)[0]
        elif isinstance(train_data, pygBatch):    # data from torch_geometric.datasets
            train_inputs = train_data.to(self.device)
            train_labels = train_data.y.to(self.device)
            train_outputs = meta_net(train_inputs)
        else:
            not NotImplementedError()
        train_loss_curr = self.criterion(train_outputs, train_labels)

        for (name, param), grad in zip(meta_net.named_parameters(), meta_grads):
            set_parameter(meta_net, name, param.add(grad, alpha=-self.eps))
        train_data = next(iter(self._dataloader(Subset(self.dataset, indices), shuffle=False)))
        if isinstance(train_data, list):          # data from torch.utils.data.Dataset
            train_inputs = train_data[0].to(self.device)
            train_labels = train_data[1].to(self.device)
            train_outputs = meta_net(train_inputs)
        elif isinstance(train_data, dict):        # data from datasets.arrow_dataset.Dataset
            train_inputs = {k: v.to(self.device) for k, v in train_data.items() 
                           if k not in ['labels', 'indices']}
            train_labels = train_data['labels'].to(self.device)
            train_outputs = meta_net(**train_inputs)[0]
        elif isinstance(train_data, pygBatch):    # data from torch_geometric.datasets
            train_inputs = train_data.to(self.device)
            train_labels = train_data.y.to(self.device)
            train_outputs = meta_net(train_inputs)
        else:
            not NotImplementedError()
        train_loss_next = self.criterion(train_outputs, train_labels)

        self.scorer.train()
        self.fc.train()
        data = next(iter(self._dataloader(Subset(self.dataset, indices), shuffle=False)))
        if isinstance(data, list):          # data from torch.utils.data.Dataset
            inputs = data[0].to(self.device)
            weights = self.fc(self.scorer(inputs)).squeeze()
        elif isinstance(data, dict):        # data from datasets.arrow_dataset.Dataset
            inputs = {k: v.to(self.device) for k, v in data.items() 
                           if k not in ['labels', 'indices']}
            weights = self.fc(self.scorer(**inputs)[0]).squeeze()
        elif isinstance(data, pygBatch):    # data from torch_geometric.datasets
            inputs = data.to(self.device)
            weights = self.fc(self.scorer(inputs)).squeeze()
        else:
            not NotImplementedError()
        self.optimizer_s.zero_grad()
        loss_s = (1.0 / self.eps * (train_loss_next - train_loss_curr)) * torch.log(weights)
        torch.mean(loss_s).backward()
        self.optimizer_s.step()
        return torch.mean(loss * weights.detach())



class DDSTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, gpu_index, num_epochs, random_seed, eps):
        
        cl = DDS(eps)

        super(DDSTrainer, self).__init__(
            data_name, net_name, gpu_index, num_epochs, random_seed, cl)