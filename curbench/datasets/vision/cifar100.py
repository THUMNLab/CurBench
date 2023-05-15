# The code is developed based on 
# https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb#file-data_loader-py
# The mean and std of CIFAR-100 is based on 
# https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151


import numpy as np
from torch.utils.data import Subset
from torchvision import datasets, transforms


def get_cifar100_dataset(data_dir='data', valid_ratio=0.1):
    assert ((valid_ratio >= 0) and (valid_ratio <= 1)), \
        'Assert Error: valid_size should be in the range [0, 1].'

    MEAN = [0.5071, 0.4865, 0.4409]
    STD = [0.2673, 0.2564, 0.2762]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_dataset = datasets.CIFAR100(
        root=data_dir, train=True, 
        download=True, transform=train_transform,
    )
    valid_dataset = datasets.CIFAR100(
        root=data_dir, train=True,
        download=True, transform=test_transform,
    )
    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False,
        download=True, transform=test_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_ratio * num_train))
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_dataset = Subset(train_dataset, train_idx)
    valid_dataset = Subset(valid_dataset, valid_idx)
    test_dataset.__setattr__('name', 'cifar100')
    test_dataset.__setattr__('num_classes', 100)
    test_dataset.__setattr__('image_size', 32)
    
    return train_dataset, valid_dataset, test_dataset
