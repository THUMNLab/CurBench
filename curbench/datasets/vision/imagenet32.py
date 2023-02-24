
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow_datasets as tfds
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import Cutout, LabelNoise


class ImageNet32(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        for image, label in self.dataset: 
            # batch_size = 1
            # image = [batch_size, height, weight, channel]
            # label = [batch_size, 1]
            if self.transform: image = self.transform(image[0])
            label = label[0]
            yield image, label

    def __len__(self):
        return len(self.dataset)


def get_imagenet32_dataset(data_dir='data', valid_ratio=0.1, 
                           augment=True, cutout_length=0, noise_ratio=0.0):
    assert ((valid_ratio >= 0) and (valid_ratio <= 1)), \
        'Assert Error: valid_size should be in the range [0, 1].'
    valid_percent = int(valid_ratio * 100)

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    train_dataset = tfds.as_numpy(tfds.load('imagenet_resized/32x32', 
        split='train[%d%%:]' % (valid_percent), data_dir=data_dir, 
        batch_size=1, shuffle_files=True, download=False, as_supervised=True)
    )
    valid_dataset, test_dataset = tfds.as_numpy(tfds.load('imagenet_resized/32x32', 
        split=['train[:%d%%]' % (valid_percent), 'validation'], data_dir=data_dir, 
        batch_size=1, shuffle_files=False, download=False, as_supervised=True)
    )
    
    transf = [
        transforms.ToPILImage(),
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

    train_dataset = ImageNet32(
        dataset=train_dataset, transform=train_transform,
    )
    valid_dataset = ImageNet32(
        dataset=valid_dataset, transform=test_transform,
    )
    test_dataset = ImageNet32(
        dataset=test_dataset, transform=test_transform,
    )

    if noise_ratio > 0.0:
        train_dataset = LabelNoise(train_dataset, noise_ratio, 1000)

    return train_dataset, valid_dataset, test_dataset


if __name__ == '__main__':
    train_dataset = tfds.as_numpy(tfds.load('imagenet_resized/32x32', 
        split='train[%d%%:]' % (0.1), data_dir='data', 
        batch_size=1, shuffle_files=True, download=False, as_supervised=True)
    )
    transform = transforms.ToPILImage()
    for image, label in train_dataset:
        print(image[0])
        print(label[0])
        print(transform(image[0]))
        input()