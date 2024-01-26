from .gcn import GCNForNode, GCNForGraph
from .gat import GATForNode, GATForGraph
from .sage import SageForNode, SageForGraph
from .gin import GIN


def get_net(net_name, dataset):
    net_dict = {
        'gcn': GCNForGraph, 
        'gat': GATForGraph, 
        'sage': SageForGraph, 
        'gin': GIN,
    }
    assert net_name in net_dict, \
        'Assert Error: net_name should be in ' + str(list(net_dict.keys()))
    
    if dataset.name in ['cora', 'citeseer', 'pubmed']:
        raise NotImplementedError()
    else:
        return net_dict[net_name](dataset)