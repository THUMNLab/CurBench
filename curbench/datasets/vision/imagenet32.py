'''
Based on https://pytorch.org/vision/stable/_modules/torchvision/datasets/cifar.html#CIFAR10
'''

import os
import pickle
from typing import Any, Callable, Optional, Tuple
import numpy as np
from PIL import Image
from torch.utils.data import Subset, Dataset
from torchvision import transforms

from .utils import Cutout


class ImageNet32(Dataset):
    """`ImageNet 32x32 <https://patrykchrabaszcz.github.io/Imagenet32/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``imagenet32`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:

        self.root = root
        self.train = train  # training set or test set
        self.transform = transform
        self.target_transform = target_transform

        if self.train:
            downloaded_list = ['train_data_batch_%d' % (i) for i in range(1, 11)]
        else:
            downloaded_list = ['val_data']

        self.data: Any = []
        self.targets = []
        self.num_classes = 1000

        # now load the picked numpy arrays
        for file_name in downloaded_list:
            file_path = os.path.join(self.root, 'imagenet32', file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                self.targets.extend(entry["labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index] - 1

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"


def get_imagenet32_dataset(data_dir='data', valid_ratio=0.1, augment=True, cutout_length=0):
    assert ((valid_ratio >= 0) and (valid_ratio <= 1)), \
        'Assert Error: valid_size should be in the range [0, 1].'
    
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    transf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ] if augment else []
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
    cutout = [Cutout(cutout_length)] \
      if cutout_length > 0 else []
    
    train_transform = transforms.Compose(transf + normalize + cutout)
    test_transform = transforms.Compose(normalize)

    train_dataset = ImageNet32('data', train=True, transform=train_transform)
    valid_dataset = ImageNet32('data', train=True, transform=test_transform)
    test_dataset = ImageNet32('data', train=False, transform=test_transform)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_ratio * num_train))
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_dataset = Subset(train_dataset, train_idx)
    valid_dataset = Subset(valid_dataset, valid_idx)

    return train_dataset, valid_dataset, test_dataset


if __name__ == '__main__':
    import torch
    from torch.utils.data import DataLoader
    import torchvision

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = ImageNet32('data', train=True, transform=transform)
    testset = ImageNet32('data', train=False, transform=transform)
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
    print(len(trainset), len(testset))

    import matplotlib.pyplot as plt
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        fig = np.transpose(npimg, (1, 2, 0))
        plt.imshow(fig)
        plt.savefig('img.png')

    for images, labels in testloader:
        imshow(torchvision.utils.make_grid(images))
        input()