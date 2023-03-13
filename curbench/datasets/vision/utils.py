import collections
import numpy as np
import torch
from torch.utils.data import Dataset


class LabelNoise(Dataset):
    def __init__(self, dataset, noise_ratio, num_labels):
        self.dataset = dataset
        self.noise_ratio = noise_ratio
        self.num_labels = num_labels
        self.labels = []
        for i, (_, y) in enumerate(self.dataset):
            if np.random.rand() < self.noise_ratio:
                self.labels.append(
                    np.random.choice(list(range(0, y)) + list(range(y + 1, num_labels))))
            else:
                self.labels.append(y)

    def __getitem__(self, index):
        return (self.dataset[index][0], self.labels[index])

    def __len__(self):
        return len(self.dataset)


class ClassImbalanced(Dataset):
    def __init__(self, dataset, mode="dominant", \
            dominant_labels=None, dominant_ratio=4, dominant_minor_floor=5,\
                exp_mu=0.9):
        self.dataset = dataset
        self.mode = mode
        self.dominant_labels = dominant_labels
        self.dominant_ratio = dominant_ratio
        self.dominant_minor_floor = dominant_minor_floor
        self.exp_mu = exp_mu
        self.idx = []
        counter = collections.Counter([self.dataset[i][1] for i in range(len(self.dataset))])

        if self.mode == "dominant":
            # the labels are classifed into dominant label and minor label
            # dominant lables are specified in `dominant_labels`, 
            # and the `dominant_ratio` denotes the ratio bettween the number of dominant class and minor class
            if not isinstance(self.dominant_labels, list) or len(list(self.dominant_labels)) < 1:
                raise ValueError("dominant_labels should be a list with at least one element")
            self.dominant_labels = np.array(self.dominant_labels)
            nums = np.array([counter[key] for key in sorted(counter.keys())])
            minor_num = max(self.dominant_minor_floor, int(max(nums[self.dominant_labels]) / self.dominant_ratio))
            minor_ratio = minor_num * 1.0 / min(nums[~self.dominant_labels])
            for i, (_, y) in enumerate(self.dataset):
                if (y not in self.dominant_labels) and (np.random.rand() > minor_ratio):
                    continue
                self.idx.append(i)
        elif self.mode == "exp":
            # the number of each class is altered according to an exponential function 
            # n = n_i * (`exp_mu` ^ i)
            # where i is the class index, and n_i is the original number for class i
            nums = np.array([counter[key] * (exp_mu ** i) for i, key in enumerate(sorted(counter.keys()))])
            for i, (_, y) in enumerate(self.dataset):
                if np.random.rand() < (nums[y] * 1.0 / counter[y]):
                    self.idx.append(i)
        elif self.mode == "none":
            self.idx = list(range(len(self.dataset)))
        else:
            raise NotImplementedError()
    
    def __getitem__(self, index):
        return self.dataset[self.idx[index]][0], self.dataset[self.idx[index]][1]
    
    def __len__(self):
        return len(self.idx)


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img



# TODO: Lighting