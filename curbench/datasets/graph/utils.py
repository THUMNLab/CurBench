import numpy as np
import torch
import collections
from torch_geometric.data import InMemoryDataset


class LabelNoise(InMemoryDataset):
    def __init__(self, dataset, noise_ratio, label_range):
        self.dataset = dataset
        self.noise_ratio = noise_ratio
        self.label_range = label_range.copy()
        if not isinstance(self.label_range, list) or len(self.label_range) != 2 \
            or self.label_range[1] < self.label_range[0]:
            raise ValueError('label range should be a list consisting of the lower and upper range')
        self.label_range[1] += 1
        gen_new_label = lambda : int(np.random.rand() * (self.label_range[1] - self.label_range[0]) + self.label_range[0])
        
        self.labels = []
        for _, data in enumerate(self.dataset):
            cur_label = data.y.item()
            if np.random.rand() < self.noise_ratio:
                new_label = gen_new_label()
                while new_label == cur_label: new_label = gen_new_label()
                self.labels.append(new_label)
            else:
                self.labels.append(cur_label)
        self.labels = torch.tensor(self.labels)
        
        self.__class__.__name__ = self.dataset.__class__.__name__
        self._indices = None
        self.slices = self.dataset.slices
        self._data_list = self.dataset._data_list
        self.transform = self.dataset.transform
    
    
    def __getitem__(self, index):
        if isinstance(index, slice):
            index = list(range(len(self.dataset)))[index]
        if isinstance(index, int):
            res = self.dataset[index]
            res.y = self.labels[index]
        else:
            res = self.dataset[index]
            for i in range(len(index)):
                res[i].y = self.labels[index[i]]
        return res
    
    def __len__(self):
        return len(self.dataset)
    
    @property
    def num_classes(self):
        return self.dataset.num_classes
    
    @property
    def num_features(self):
        return self.dataset.num_features
    
    
class ClassImbalanced(InMemoryDataset):
    def __init__(self, dataset, mode='dominant', \
        dominant_labels=None, dominant_ratio=4, dominant_minor_floor=5,\
                exp_mu=0.9):
        self.dataset = dataset
        self.mode = mode
        self.dominant_labels = dominant_labels
        self.dominant_ratio = dominant_ratio
        self.dominant_minor_floor = dominant_minor_floor
        self.exp_mu = exp_mu
        self.idx = []
        counter = collections.Counter([self.dataset.data.y[i].item() for i in range(len(self.dataset.data.y))])
        if self.mode == 'dominant':
            if not isinstance(self.dominant_labels, list) or len(list(self.dominant_labels)) < 1:
                raise ValueError("dominant_labels should be a list with at least one element")
            self.dominant_labels = np.array(self.dominant_labels)
            nums = np.array([counter[key] for key in sorted(counter.keys())])
            minor_num = max(self.dominant_minor_floor, int(max(nums[self.dominant_labels]) / self.dominant_ratio))
            minor_ratio = minor_num * 1.0 / min(nums[~self.dominant_labels])
            for i, data in enumerate(self.dataset):
                if (data.y.item() not in self.dominant_labels) and (np.random.rand() > minor_ratio):
                    continue
                self.idx.append(i)
        elif self.mode == 'exp':
            nums = np.array([counter[key] * (exp_mu ** i) for i, key in enumerate(sorted(counter.keys()))])
            for i, data in enumerate(self.dataset):
                label = data.y.item()
                if np.random.rand() < (nums[label] * 1.0 / counter[label]):
                    self.idx.append(i)
        elif self.mode == 'none':
            self.idx = list(range(len(self.dataset)))
        else:
            raise NotImplementedError()
        self.idx = np.array(self.idx)
        
        self.__class__.__name__ = self.dataset.__class__.__name__
        self._indices = None
        self.slices = self.dataset.slices
        self._data_list = self.dataset._data_list
        self.transform = self.dataset.transform
    
    def __getitem__(self, index):
        return self.dataset[self.idx[index]]
        
    def __len__(self):
        return len(self.idx)
    
    @property
    def num_classes(self):
        return self.dataset.num_classes
    
    @property
    def num_features(self):
        return self.dataset.num_features