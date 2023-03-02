import numpy as np
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.transforms import NormalizeFeatures


name_trans = {
    'cora': 'Cora', 'citeseer': 'CiteSeer', 'pubmed': 'PubMed',
    'mutag': 'MUTAG', 'nci1': 'NCI1', 'proteins': 'PROTEINS', 
    'collab': 'COLLAB', 'dd': 'DD', 'ptc_mr': 'PTC_MR', 'imdb-binary': 'IMDB-BINARY',
}

def get_dataset(data_name):
    if data_name in name_trans: 
        data_name = name_trans[data_name]

    if data_name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root='data', name=data_name, split='full', transform=NormalizeFeatures())
    elif data_name in ['MUTAG', 'NCI1', 'PROTEINS', 'COLLAB', 'DD', 'PTC_MR', 'IMDB-BINARY']:
        dataset = TUDataset(root='data', name=data_name)
    else:
        raise NotImplementedError()
    return dataset


def split_dataset(dataset, split=[8, 1, 1]):
    split = np.array(split)
    assert (split > 0).all(), 'Assert Error: split ratio should be positive'

    split = split.cumsum() / split.sum()
    idx_valid = int(split[0] * len(dataset))
    idx_test = int(split[1] * len(dataset))

    dataset = dataset.shuffle()
    return dataset[:idx_valid], dataset[idx_valid:idx_test], dataset[idx_test:]