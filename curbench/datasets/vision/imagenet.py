'''
Based on https://github.com/jiweibo/ImageNet/blob/master/data_loader.py
'''


import os
import pickle
import numpy as np

from torch.utils.data import Dataset, random_split
from torchvision import transforms


class ImageNet(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = root
        self.train = train  # training set or test set
        self.transform = transform
        self.target_transform = target_transform

        file_name = 'train_batch' if self.train else 'val_batch'
        file_path = os.path.join(self.root, 'imagenet-1k', file_name)

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


def get_imagenet_dataset(data_dir='data', valid_ratio=0.1):
    assert ((valid_ratio >= 0) and (valid_ratio <= 1)), \
        'Assert Error: valid_size should be in the range [0, 1].'
    
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(224),
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

    train_dataset = ImageNet(data_dir, train=True, transform=train_transform)
    test_dataset = ImageNet(data_dir, train=False, transform=test_transform)

    num_train = len(train_dataset)
    num_valid = int(np.floor(valid_ratio * num_train))
    train_dataset, valid_dataset = random_split(train_dataset, [num_train - num_valid, num_valid])

    test_dataset.__setattr__('name', 'imagenet')
    test_dataset.__setattr__('num_classes', 1000)
    test_dataset.__setattr__('image_size', 224)

    return train_dataset, valid_dataset, test_dataset
