import os
import pickle
import numpy as np

from torch.utils.data import Subset, Dataset
from torchvision import transforms


class Animal10N(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = root
        self.train = train  # training set or test set
        self.transform = transform
        self.target_transform = target_transform

        file_name = 'train_batch' if self.train else 'val_batch'
        file_path = os.path.join(self.root, 'animal-10n', file_name)

        with open(file_path, 'rb') as f:
            entry = pickle.load(f)
            self.data = entry['data']
            self.targets = entry['labels']

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def get_animal10n_dataset(data_dir='data', valid_ratio=0.1):
    assert ((valid_ratio >= 0) and (valid_ratio <= 1)), \
        'Assert Error: valid_size should be in the range [0, 1].'
    
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_dataset = Animal10N(data_dir, train=True, transform=train_transform)
    valid_dataset = Animal10N(data_dir, train=True, transform=test_transform)
    test_dataset = Animal10N(data_dir, train=False, transform=test_transform)

    for dataset in [train_dataset, valid_dataset, test_dataset]:
        dataset.__setattr__('name', 'animal10n')
        dataset.__setattr__('num_classes', 10)
        dataset.__setattr__('image_size', 64)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_ratio * num_train))
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_dataset = Subset(train_dataset, train_idx)
    valid_dataset = Subset(valid_dataset, valid_idx)

    return train_dataset, valid_dataset, test_dataset
