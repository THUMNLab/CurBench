from .cifar10 import get_cifar10_dataset
from .cifar100 import get_cifar100_dataset
from .imagenet32 import get_imagenet32_dataset
from .tinyimagenet import get_tinyimagenet_dataset
from .utils import LabelNoise, LabelImbalance


data_dict = {
    'cifar10': get_cifar10_dataset,
    'cifar100': get_cifar100_dataset,
    'imagenet32': get_imagenet32_dataset,
    'tinyimagenet': get_tinyimagenet_dataset,
}


def get_dataset(data_name):
    assert not ('noise' in data_name and 'imbalance' in data_name), \
        'Assert Error: only support one setting from [standard, noise, imbalance]'

    # noise setting
    if 'noise' in data_name:
        try:
            parts = data_name.split('-')
            data_name = parts[0]
            noise_ratio = float(parts[-1])
        except:
            assert False, 'Assert Error: data_name shoule be [dataset]-noise-[ratio]'
        assert noise_ratio >= 0.0 and noise_ratio <= 1.0, \
            'Assert Error: noise ratio should be in range of [0.0, 1.0]'
    else:
        noise_ratio = None

    # imbalance setting
    if 'imbalance' in data_name:
        try:
            parts = data_name.split('-')
            data_name = parts[0]
            imbalance_ratio = int(parts[-1])
        except:
            assert False, 'Assert Error: data_name shoule be [dataset]-imbalance-[ratio]'
        assert imbalance_ratio >= 1 and imbalance_ratio <= 200, \
            'Assert Error: imbalance ratio should be in range of [1, 200]'
    else:
        imbalance_ratio = None

    # get standard, noisy or imbalanced dataset
    assert data_name in data_dict, 'Assert Error: data_name should be in ' + str(list(data_dict.keys()))
    train_dataset, valid_dataset, test_dataset = data_dict[data_name]()
    if noise_ratio: 
        train_dataset = LabelNoise(train_dataset, noise_ratio, test_dataset.num_classes)
    if imbalance_ratio: 
        train_dataset = LabelImbalance(train_dataset, imbalance_ratio, test_dataset.num_classes)
    
    return train_dataset, valid_dataset, test_dataset
