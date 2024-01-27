from .gcn import GCNForGraph
from .gat import GATForGraph
from .sage import SageForGraph
from .gin import GIN
from .gnn_ogb import GNNForOGB


def get_net(net_name, dataset):
    if dataset.name in ['cora', 'citeseer', 'pubmed']:
        raise NotImplementedError()
    elif dataset.name in ['ogbg-molhiv']:
        return GNNForOGB(dataset.num_classes, gnn_type=net_name)
    else: # TUDataset
        net_dict = {
            'gcn': GCNForGraph, 
            'gat': GATForGraph, 
            'sage': SageForGraph, 
            'gin': GIN,
        }
        assert net_name in net_dict, \
            'Assert Error: net_name should be in ' + str(list(net_dict.keys()))
        return net_dict[net_name](dataset)