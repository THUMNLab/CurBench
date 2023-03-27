import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.data.batch import Batch as pygBatch

from .base import BaseTrainer, BaseCL



class Minimax(BaseCL):
    """Minimax CL Algorithm.
    
    Minimax curriculum learning: Machine teaching with desirable difficulties and scheduled diversity. https://openreview.net/pdf?id=BywyFQlAW
    """
    def __init__(self, schedule_epoch, warm_epoch, lam, minlam, gamma, delta,
                 initial_size, fe_alpha, fe_beta, fe_gamma, fe_lambda,
                 fe_entropy, fe_gsrow, fe_central_op, fe_central_min, fe_central_sum):
        super(Minimax, self).__init__()

        self.name = 'minimax'
        self.epoch = 0
        self.schedule_epoch = schedule_epoch
        self.warm_epoch = warm_epoch
        self.lam = lam
        self.minlam = minlam
        self.gamma = gamma
        self.cnt = 0
        self.initial_size = initial_size
        self.delta = delta
        self.fe_alpha = fe_alpha
        self.fe_beta = fe_beta
        self.fe_gamma = fe_gamma
        self.fe_lambda = fe_lambda
        self.fe_entropy = fe_entropy
        self.fe_gsrow = fe_gsrow
        self.fe_central_op = fe_central_op
        self.fe_central_min = fe_central_min
        self.fe_central_sum = fe_central_sum
    

    def data_prepare(self, loader, **kwargs):
        super().data_prepare(loader)
        self.dataloader = loader
        
        if self.initial_size is None:
            self.siz = 0.1 * self.data_size
        else:
            self.siz = self.initial_size * self.data_size
        self.loss = np.zeros(self.data_size)
        self.features = np.zeros(self.data_size)
        self.centrality = np.zeros(self.data_size)
        self.train_set = np.arange(self.data_size)


    def model_prepare(self, net, device, epochs, criterion, optimizer, lr_scheduler, **kwargs):
        super().model_prepare(net, device, epochs, criterion, optimizer, lr_scheduler)
        if self.delta is None:
            self.delta = int((self.data_size - self.siz) / (int(self.epochs / self.schedule_epoch)))
        else:
            self.delta = self.delta * self.data_size

        self.num_classes = self.net.num_labels


    def data_curriculum(self, **kwargs):
        if self.epoch % self.schedule_epoch == 0:
            if self.epoch != 0:
                self.lam = max(self.lam * (1 - self.gamma), self.minlam)
                self.siz = min(self.siz + self.delta, self.data_size)
            # at the begining of each episode, train a neural network for few epochs to calculate the features used in the following MCL steps
            dataloader_ = self._dataloader(self.dataset, shuffle=False)
            self._pretrain(dataloader_)
            self.features = self._pretest(dataloader_)
            self.features = np.clip(self.features, 1e-10, 1e10)
            self.centrality = self._compute_centrality(self.features, self.fe_alpha, self.fe_beta, self.fe_gamma, self.fe_lambda, \
                                                    self.fe_gsrow, self.fe_entropy, self.fe_central_op, self.fe_central_min, self.fe_central_sum)

        if self.epoch < self.warm_epoch:
            dataloader = self._dataloader(self.dataset, shuffle=False)
        else:
            pro = self.loss + self.lam * self.centrality
            pro = pro / np.sum(pro)
            self.train_set = np.random.choice(self.data_size, int(self.siz), p=pro, replace=False).tolist()
            dataset = Subset(self.dataset, self.train_set)
            dataloader = self._dataloader(dataset, shuffle=False)

        self.epoch += 1
        self.cnt = 0
        return dataloader
    

    def loss_curriculum(self, outputs, labels, indices, **kwargs):
        losses = self.criterion(outputs, labels)
        for loss in losses:
            self.loss[self.train_set[self.cnt]] = loss
            self.cnt += 1
        return torch.mean(losses)
    

    def _pretrain(self, dataloader):
        self.net.train()
        for step, data in enumerate(dataloader):
            if isinstance(data, list):  
                # image classification
                inputs = data[0].to(self.device)
                labels = data[1].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels).mean()
                loss.backward()
            elif isinstance(data, dict): 
                # text classification
                inputs = {k: v.to(self.device) for k, v in data.items() 
                            if k not in ['labels', 'indices']}
                labels = data['labels'].to(self.device)
                indices = data['indices'].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(**inputs)[0] # logits, (hidden_states), (attentions)
                loss = self.criterion(outputs, labels).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
            elif isinstance(data, pygBatch):
                inputs = data.to(self.device)
                labels = data.y.to(self.device)
                indices = data.i.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels).mean()   # curriculum part
                loss.backward()
            else:
                raise NotImplementedError()
            self.optimizer.step()
        self.lr_scheduler.step()
    

    def _pretest(self, dataloader):
        all_feature = np.array([])
        _net = copy.deepcopy(self.net)
        _net.fc = nn.Identity()
        _net.eval()
        for step, data in enumerate(dataloader):
            if isinstance(data, list):
                inputs = data[0].to(self.device)
                with torch.no_grad():
                    feature = _net(inputs)
            elif isinstance(data, dict):
                # text classification
                inputs = {k: v.to(self.device) for k, v in data.items() 
                            if k not in ['labels', 'indices']}
                with torch.no_grad():
                    feature = self.net(**inputs)[0]
            elif isinstance(data, pygBatch):
                inputs = data.to(self.device)
                with torch.no_grad():
                    feature = self.net(inputs)
            else:
                raise NotImplementedError()
            all_feature = np.append(all_feature, feature.cpu())
        all_feature = all_feature.reshape(int(self.data_size), int(len(all_feature) / self.data_size))
        return all_feature
    

    def _entropy(self, labels, base=None):
        num_labels = len(labels)
        value, count = np.unique(labels, return_counts=True)
        prob = count / num_labels
        num_classes = np.count_nonzero(prob)
        if num_labels <= 1 or num_classes <= 1:
            return 1
        entro = 0
        if base == None:
            base = np.e 
        for iter in prob:
            entro -= iter * math.log(iter, base)
        return entro
    

    def _swarp(self, feature, alpha, beta):
        swarp_epsilon = 1e-30
        nonzero_indice = (feature > swarp_epsilon)
        if np.any(nonzero_indice) > 0:
            feature[nonzero_indice] = 1.0 / (1 + (feature[nonzero_indice] ** (1 / np.log2(beta)) - 1) ** alpha)
        return feature


    def _compute_centrality(self, feature, fe_alpha, fe_beta, fe_gamma, fe_lambda, fe_gsmin, fe_entropy, fe_central_op, fe_central_min, fe_central_sum):
        # First process the feature matrix, using gamma correction
        if fe_gamma != 1.0 or fe_alpha != 1.0 or fe_beta != 0.5:
            if fe_gsmin == True:
                feature_min = feature.min(axis=0)
                feature_max = feature.max(axis=0) + 1e-5
                feature = feature - feature_min
                feature = feature / feature_max
            else:
                feature_max = feature.max(axis=0)
                feature = feature / feature_max
        
        if fe_gamma != 1.0:
            if fe_alpha != 1.0 or fe_beta != 0.5:
                feature = self._swarp(feature, fe_alpha, fe_beta) ** fe_gamma
            else:
                feature = feature ** fe_gamma
        feature = feature * feature_max

        # Then compute the centrality
        centrality = None
        if fe_lambda < 1.0:
            centrality = np.zeros(feature.shape[0])
            if fe_entropy is True:
                max_entropy = np.log2(feature.shape[1])
                for index in range(feature.shape[0]):
                    centrality[index] = max_entropy - self._entropy(feature[index].T)
                centrality = centrality / np.sum(centrality)
            else:
                if fe_central_op is True:
                    centrality = np.sum(feature.dot(feature.transpose()), axis=1)
                    centrality = centrality / np.sum(centrality)
                elif fe_central_min is True:
                    centrality = np.min(feature, axis=1)
                    centrality = centrality / np.sum(centrality)
                elif fe_central_sum is True:
                    centrality = np.sum(feature, axis=1)
                    centrality = centrality / np.sum(centrality)
        return centrality


class MinimaxTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, gpu_index, num_epochs, random_seed,
                 schedule_epoch, warm_epoch, lam, minlam, gamma, delta,
                 initial_size, fe_alpha, fe_beta, fe_gamma, fe_lambda,
                 fe_entropy, fe_gsrow, fe_central_op, fe_central_min, fe_central_sum):
        
        cl = Minimax(schedule_epoch, warm_epoch, lam, minlam, gamma, delta,
                 initial_size, fe_alpha, fe_beta, fe_gamma, fe_lambda,
                 fe_entropy, fe_gsrow, fe_central_op, fe_central_min, fe_central_sum)

        super(MinimaxTrainer, self).__init__(
            data_name, net_name, gpu_index, num_epochs, random_seed, cl)
