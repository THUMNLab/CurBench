import copy
import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.data.batch import Batch as pygBatch

from .base import BaseTrainer, BaseCL
from .utils import VNet, set_parameter



class MetaWeightNet(BaseCL):
    """Meta-Weight-Net CL Algorithm.
    
    Meta-weight-net: Learning an explicit mapping for sample weighting. https://proceedings.neurips.cc/paper/2019/file/e58cc5ca94270acaceed13bc82dfedf7-Paper.pdf
    """
    def __init__(self, ):
        super(MetaWeightNet, self).__init__()
        self.name = 'meta_weight_net'


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
        self.vnet = VNet(1, 100, 1).to(self.device)
        self.optimizer_v = torch.optim.Adam(self.vnet.parameters(), lr=1e-3, weight_decay=1e-4)


    def data_curriculum(self, **kwargs):
        return self._dataloader(self.train_dataset, shuffle=True)


    def loss_curriculum(self, outputs, labels, indices, **kwargs):
        if not isinstance(indices, list): indices = indices.tolist()
        
        meta_net = copy.deepcopy(self.net)
        meta_net.train()
        meta_net.zero_grad()
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
        train_loss = self.criterion(train_outputs, train_labels).view(-1, 1)
        v_lambda = self.vnet(train_loss)
        grads = torch.autograd.grad(torch.mean(train_loss * v_lambda), (meta_net.parameters()), retain_graph=True, create_graph=True)
        for (name, param), grad in zip(meta_net.named_parameters(), grads):
            set_parameter(meta_net, name, param.add(grad, alpha=-self.optimizer.param_groups[0]['lr']))

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
        self.optimizer_v.zero_grad()
        torch.mean(meta_loss).backward()
        self.optimizer_v.step()

        loss = self.criterion(outputs, labels).view(-1, 1)
        weights = self.vnet(loss)
        return torch.mean(loss * weights.detach())


class MetaWeightNetTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, gpu_index, num_epochs, random_seed):
        
        cl = MetaWeightNet()

        super(MetaWeightNetTrainer, self).__init__(
            data_name, net_name, gpu_index, num_epochs, random_seed, cl)