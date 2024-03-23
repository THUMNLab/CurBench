import numpy as np
from torch.utils.data import Dataset


class LabelNoise(Dataset):
    def __init__(self, dataset, noise_ratio, num_classes):
        self.dataset = dataset

        self.labels = []
        noise_cnt = 0
        for _, y in self.dataset:
            if np.random.rand() < noise_ratio:
                noise_cnt += 1
                self.labels.append(
                    np.random.choice(list(range(0, y)) + list(range(y + 1, num_classes))))
            else:
                self.labels.append(y)
        print('Noise label = %d / %d' % (noise_cnt, len(self.dataset)))

    def __getitem__(self, i):
        return (self.dataset[i][0], self.labels[i])

    def __len__(self):
        return len(self.dataset)


class LabelImbalance(Dataset):
    def __init__(self, dataset, imbalance_ratio, num_classes):
        '''
        the number of each class is altered according to an exponential function 
        n = n_i * (mu ^ i)
        where i is the class index, and n_i is the original number for class i
        '''
        self.dataset = dataset
        self.indices = []

        label_cnts = [0] * num_classes
        for i, (_, y) in enumerate(self.dataset):
            label_cnts[y] += 1
        print('Original  label: ', np.array(label_cnts))

        label_cnts = [0] * num_classes
        mu = (1.0 / imbalance_ratio) ** (1.0 / (num_classes - 1))
        for i, (_, y) in enumerate(self.dataset):
            if np.random.rand() < (mu ** y):
                label_cnts[y] += 1
                self.indices.append(i)
        print('Imbalance label: ', np.array(label_cnts))
        
    def __getitem__(self, i):
        return self.dataset[self.indices[i]]
    
    def __len__(self):
        return len(self.indices)
