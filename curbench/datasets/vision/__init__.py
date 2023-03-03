from .cifar10 import get_cifar10_dataset
from .cifar100 import get_cifar100_dataset
from .imagenet32 import get_imagenet32_dataset


data_dict = {
    'cifar10': get_cifar10_dataset,
    'cifar100': get_cifar100_dataset,
    'imagenet32': get_imagenet32_dataset,
}


def get_dataset(data_name):
    assert data_name in data_dict, \
        'Assert Error: data_name should be in ' + str(list(data_dict.keys()))
    return data_dict[data_name]()


def get_dataset_with_noise(data_name):
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
        noise_ratio = 0.0

    assert data_name in data_dict, \
        'Assert Error: data_name should be in ' + str(list(data_dict.keys()))
    return data_dict[data_name](noise_ratio=noise_ratio)
