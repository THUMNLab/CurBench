from .gcn import GCN4Node, GCN4Graph
from .gat import GAT4Node, GAT4Graph
from .sage import Sage4Node, Sage4Graph


def get_net(net_name, dataset):
    net_set = set(['gcn', 'gat', 'sage'])
    assert net_name in net_set, \
        'Assert Error: net_name should be in ' + str(list(net_set))
    
    dataset_name = dataset.__class__.__name__
    net_name = net_name.upper() + '4' + dataset_name
    net_dict = {
        'GCN4Planetoid': GCN4Node,
        'GCN4TUDataset': GCN4Graph,
        'GAT4Planetoid': GAT4Node,
        'GAT4TUDataset': GAT4Graph,
        'GAGE4Planetoid': Sage4Node,
        'GAGE4TUDataset': Sage4Graph,
    }
    return net_dict[net_name](dataset)