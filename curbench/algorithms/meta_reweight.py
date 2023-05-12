import copy
import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.data.batch import Batch as pygBatch

from .base import BaseTrainer, BaseCL
from .utils import set_parameter



class MetaReweight(BaseCL):
    """Meta Reweight CL Algorithm.
    
    Learning to reweight examples for robust deep learning. http://proceedings.mlr.press/v80/ren18a/ren18a.pdf
    """
    def __init__(self, ):
        super(MetaReweight, self).__init__()
        self.name = 'meta_reweight'
    

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
        train_loss = self.criterion(train_outputs, train_labels)
        eps = torch.zeros_like(train_loss, requires_grad=True)
        grads = torch.autograd.grad(torch.mean(train_loss * eps), (meta_net.parameters()), retain_graph=True, create_graph=True)
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
        grad_eps = torch.autograd.grad(torch.mean(meta_loss), eps)[0]

        w_tilde = torch.clamp(-grad_eps, min=0.0)
        norm_c = torch.sum(w_tilde) + 1e-12
        weights = w_tilde / norm_c
        loss = self.criterion(outputs, labels)
        return torch.mean(loss * weights.detach())


class MetaReweightTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, gpu_index, num_epochs, random_seed):
        
        cl = MetaReweight()

        super(MetaReweightTrainer, self).__init__(
            data_name, net_name, gpu_index, num_epochs, random_seed, cl)