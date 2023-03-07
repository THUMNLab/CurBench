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


def get_dataset_with_imbalanced_class(data_name):
    if 'imbalance' in data_name:
        try:
            parts = data_name.split('-')
            data_name = parts[0]
            imbalance_mode = parts[2]
            imbalance_dominant_labels = list(eval(parts[3]))
            imbalance_dominant_ratio = float(parts[4])
            imbalance_dominant_minor_floor = int(parts[5])
            imbalance_exp_mu = float(parts[6])
        except:
            assert False, 'Assert Error: data_name shoule be [dataset]-imbalance-[mode]-[dominant_labels]-[dominant_ratio]-[dominant_minor_floor]-[exp_mu]'
    else:
        imbalance_mode = 'none'
        imbalance_dominant_labels = None
        imbalance_dominant_ratio = 1
        imbalance_dominant_minor_floor = 0
        imbalance_exp_mu = 1
    
    assert data_name in data_dict, \
        'Assert Error: data_name should be in ' + str(list(data_dict.keys()))

    return data_dict[data_name](imbalance_mode=imbalance_mode, imbalance_dominant_labels=imbalance_dominant_labels, imbalance_dominant_ratio=imbalance_dominant_ratio,\
        imbalance_dominant_minor_floor=imbalance_dominant_minor_floor, imbalance_exp_mu=imbalance_exp_mu)