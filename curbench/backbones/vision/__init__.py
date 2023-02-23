from .lenet import LeNet
from .vgg import VGG16
from .resnet import ResNet18


def get_net(net_name, data_name):
    net_dict = {
        'lenet': LeNet,
        'vgg16': VGG16,
        'resnet18': ResNet18,
    }

    assert net_name in net_dict, \
        'Assert Error: net_name should be in ' + str(list(net_dict.keys()))

    classes_dict = {
        'cifar10': 10, 
        'cifar100': 100,
    }
    
    return net_dict[net_name](num_classes=classes_dict[data_name])