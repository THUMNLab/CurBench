from .gcn import GCNForNode, GCNForGraph
from .gat import GATForNode, GATForGraph
from .sage import SageForNode, SageForGraph


def get_net(net_name, dataset):
    net_set = set(['gcn', 'gat', 'sage'])
    assert net_name in net_set, \
        'Assert Error: net_name should be in ' + str(list(net_set))
    
    dataset_name = dataset.__class__.__name__
    net_name = net_name.upper() + 'For' + dataset_name
    net_dict = {
        'GCNForPlanetoid': GCNForNode,
        'GCNForTUDataset': GCNForGraph,
        'GATForPlanetoid': GATForNode,
        'GATForTUDataset': GATForGraph,
        'GAGEForPlanetoid': SageForNode,
        'GAGEForTUDataset': SageForGraph,
    }
    return net_dict[net_name](dataset)