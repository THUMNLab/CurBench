from .gcn import GCN


def get_net(net_name, dataset):
    net_dict = {
        'gcn': GCN
    }

    assert net_name in net_dict, \
        'Assert Error: net_name should be in ' + str(list(net_dict.keys()))
    
    return net_dict[net_name](dataset)