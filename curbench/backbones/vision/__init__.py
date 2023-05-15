from .lenet import LeNet
from .resnet import ResNet18
from .vit import ViT


def get_net(net_name, dataset):
    net_dict = {
        'lenet': LeNet,
        'resnet18': ResNet18,
        'vit': ViT,
    }
    assert net_name in net_dict, \
        'Assert Error: net_name should be in ' + str(list(net_dict.keys()))
    
    _, _, test_dataset = dataset
    net = net_dict[net_name](
        num_classes=test_dataset.num_classes,
        image_size=test_dataset.image_size, 
        patch_size=test_dataset.image_size // 8,
    )
    net.__setattr__('name', net_name)
    net.__setattr__('num_classes', test_dataset.num_classes)
    return net
