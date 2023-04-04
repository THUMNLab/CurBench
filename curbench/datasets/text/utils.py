import numpy as np
import torch
import torch.nn.functional as F
import collections
import datasets
from datasets import Dataset
from datasets.arrow_dataset import Dataset as SubDataset


class SlicingDataset(SubDataset):
    def __init__(self, dataset, idx):
        self.idx = idx
        self.dataset = dataset
        
    def __getitem__(self, index):
        return self.dataset[self.idx[index].tolist()]
        
    def __len__(self):
        return len(self.idx)

class SplitedDataset(Dataset):
    def __init__(self, dataset, ratio=0.95):
        self.dataset = dataset
        self.ratio = ratio
        self.train_len = len(self.dataset['train'])
        self.train_idx = np.array(range(self.train_len))
        self.test_idx = np.random.choice(a=self.train_idx, size=int(self.train_len * (1 - ratio)), replace=False)
        self.train_idx = np.delete(self.train_idx, self.test_idx)
        self.train_dataset = SlicingDataset(self.dataset['train'], self.train_idx)
        self.test_dataset = SlicingDataset(self.dataset['train'], self.test_idx)
        
    def __getitem__(self, key):
        if key == 'train':
            return self.train_dataset
        elif key == 'test':
            return self.test_dataset
        else:
            return self.dataset[key]

class UtilDataset(SubDataset):
    def __init__(self, dataset, idx, labels, use_labels=True):
        self.dataset = dataset
        self.idx = idx
        self.labels = torch.tensor(labels)
        self.use_labels = use_labels
        self.keys = self.dataset[0].keys()
        
    def __getitem__(self, index):
        res = self.dataset[self.idx[index]]    
        res['labels'] = self.labels[self.idx[index]]
        return res    

    def __getitems__(self, indexs):
        res = [{
            key: self.dataset[self.idx[index]][key] 
            if (self.use_labels is False or key != 'labels') else self.labels[self.idx[index]]
            for key in self.keys
        } for index in indexs]

        # pad the entries in each data dict to achieve equal length for all
        entries = ['input_ids', 'token_type_ids', 'attention_mask']
        for entry in entries:
            max_len_input_ids = max([len(res[i][entry]) for i in range(len(res))])
            for i in range(len(res)):
                res[i][entry] = F.pad(res[i][entry], (0, max_len_input_ids - len(res[i][entry])), 'constant', 0)
        return res
    
    def __len__(self):
        return len(self.idx)


class LabelNoise(Dataset):
    def __init__(self, dataset, noise_ratio, label_range, label_int=True):
        self.dataset = dataset
        self.noise_ratio = noise_ratio
        self.label_range = label_range.copy()
        self.label_int = label_int
        if not isinstance(self.label_range, list) or len(self.label_range) != 2 \
            or self.label_range[1] < self.label_range[0]:
            raise ValueError('label range should be a list consisting of the lower and upper range')
        if self.label_int:
            self.label_range[1] += 1
        # generate a new label in range [label_range[0], label_range[1]]
        _gen_new_label = lambda : np.random.rand() * (self.label_range[1] - self.label_range[0])
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
        self.train_subdataset = UtilDataset(self.dataset['train'], self.idx, self.labels, False)
        
    def __getitem__(self, key):
        if key == 'train':
            return self.train_subdataset
        else:
            return self.dataset[key]
        
    def __len__(self):
        return len(self.dataset)


class ClassImbalanced(Dataset):
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
        counter = collections.Counter([self.dataset['train'][i]['labels'].item() for i in range(len(self.dataset['train']))])
        if self.mode == 'dominant':
            if not isinstance(self.dominant_labels, list) or len(list(self.dominant_labels)) < 1:
                raise ValueError("dominant_labels should be a list with at least one element")
            self.dominant_labels = np.array(self.dominant_labels)
            nums = np.array([counter[key] for key in sorted(counter.keys())])
            minor_num = max(self.dominant_minor_floor, int(max(nums[self.dominant_labels]) / self.dominant_ratio))
            minor_ratio = minor_num * 1.0 / min(nums[~self.dominant_labels])
            for i, data in enumerate(self.dataset['train']):
                if (data['labels'].item() not in self.dominant_labels) and (np.random.rand() > minor_ratio):
                    continue
                self.idx.append(i)
        elif self.mode == 'exp':
            nums = np.array([counter[key] * (exp_mu ** i) for i, key in enumerate(sorted(counter.keys()))])
            for i, data in enumerate(self.dataset['train']):
                label = data['labels'].item()
                if np.random.rand() < (nums[label] * 1.0 / counter[label]):
                    self.idx.append(i)
        elif self.mode == 'none':
            self.idx = list(range(len(self.dataset['train'])))
        else:
            raise NotImplementedError()
        self.train_subdataset = UtilDataset(self.dataset['train'], self.idx, [], False)
                                     
    def __getitem__(self, key):
        if key == 'train':
            return self.train_subdataset
        else:
            return self.dataset[key]
    
    def __len__(self):
        return len(self.dataset)
        