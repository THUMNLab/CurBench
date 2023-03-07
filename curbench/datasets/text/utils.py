import numpy as np
import torch
from datasets import Dataset
from datasets.arrow_dataset import Dataset as SubDataset


class UtilDataset(SubDataset):
    def __init__(self, dataset, idx, labels):
        self.dataset = dataset
        self.idx = np.array(idx)
        self.labels = torch.tensor(labels)
        self.keys = self.dataset[0].keys()
        
    def __getitems__(self, indexs):
        res = [{
            key: self.dataset[int(self.idx[index])][key] 
            if key != 'labels' else self.labels[int(self.idx[index])]
            for key in self.keys
        } for index in indexs]
        return res
    
    def __len__(self):
        return len(self.idx)


class LabelNoise(Dataset):
    def __init__(self, dataset, noise_ratio, label_range, label_int=True):
        self.dataset = dataset
        self.noise_ratio = noise_ratio
        self.label_range = label_range
        self.label_int = label_int
        if not isinstance(self.label_range, list) or len(self.label_range) != 2 \
            or self.label_range[1] < self.label_range[0]:
            raise ValueError('label range should be a list consisting of the lower and upper range')
        if self.label_int:
            self.label_range[1] += 1
        # generate a new label in range [label_range[0], label_range[1]]
        _gen_new_label = lambda : np.random.random() * (self.label_range[1] - self.label_range[0])
        gen_new_label = lambda : (int(_gen_new_label()) + self.label_range[0]) if self.label_int else (_gen_new_label() + self.label_range[0])

        self.labels = []
        self.idx = list(range(len(self.dataset['train'])))
        for _, data in enumerate(self.dataset['train']):
            cur_label = data['labels'].item()
            if np.random.rand() < self.noise_ratio:
                new_label = gen_new_label()
                while new_label == cur_label: new_label = gen_new_label()
                self.labels.append(torch.tensor(new_label))
            else:
                self.labels.append(data['labels'])
        self.train_subdataset = UtilDataset(self.dataset['train'], self.idx, self.labels)
        
    def __getitem__(self, key):
        if key == 'train':
            return self.train_subdataset
        else:
            return self.dataset[key]
        
    def __len__(self):
        return len(self.dataset)


class ClassImbalanced(Dataset):
    def __init__(self):
        pass