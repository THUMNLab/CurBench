'''
Based on https://github.com/jiweibo/ImageNet/blob/master/data_loader.py
'''


import os
import numpy as np

from torch.utils.data import Subset
from torchvision import transforms, datasets


def get_imagenet_dataset(data_dir='data', valid_ratio=0.1):
    assert ((valid_ratio >= 0) and (valid_ratio <= 1)), \
        'Assert Error: valid_size should be in the range [0, 1].'
    
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'imagenet-1k', 'train'), 
        transform=train_transform,
    )
    valid_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'imagenet-1k', 'train'), 
        transform=test_transform,
    )
    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'imagenet-1k', 'val'), 
        transform=test_transform,
    )

    for dataset in [train_dataset, valid_dataset, test_dataset]:
        dataset.__setattr__('name', 'imagenet')
        dataset.__setattr__('num_classes', 1000)
        dataset.__setattr__('image_size', 224)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_ratio * num_train))
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_dataset = Subset(train_dataset, train_idx)
    valid_dataset = Subset(valid_dataset, valid_idx)

    return train_dataset, valid_dataset, test_dataset
