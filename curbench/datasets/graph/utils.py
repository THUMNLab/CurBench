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
    def __init__(self, dataset, imbalance_ratio):
        self.dataset = dataset
        self.idx = []
        self.imbalance_ratio = imbalance_ratio

        counter = collections.Counter([data.y.item() for i, data in enumerate(self.dataset)])
        label_cnts = [counter[key] for key in sorted(counter.keys())]
        print('Original label: ', np.array(label_cnts))

        label_cnts = [0 for key in counter.keys()]
        mu = (1.0 / imbalance_ratio) ** (1.0 / (len(counter.keys()) - 1))
        for i, data in enumerate(self.dataset):
            label = data.y.item()
            if np.random.rand() < (mu ** label) * counter[0] / counter[label]:
                label_cnts[label] += 1
                self.idx.append(i)
        print('Imbalance label: ', np.array(label_cnts))
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