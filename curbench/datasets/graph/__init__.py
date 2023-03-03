from .planetoid import get_planetoid_dataset
from .tudataset import get_tudataset_dataset, split_dataset


def get_dataset(data_name):
    name_trans = {
        'cora': 'Cora', 'citeseer': 'CiteSeer', 'pubmed': 'PubMed',
        'mutag': 'MUTAG', 'nci1': 'NCI1', 'proteins': 'PROTEINS', 
        'collab': 'COLLAB', 'dd': 'DD', 'ptc_mr': 'PTC_MR', 'imdb-binary': 'IMDB-BINARY',
    }

    if data_name in name_trans: 
        data_name = name_trans[data_name]

    if data_name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = get_planetoid_dataset('data/planetoid', data_name)
    elif data_name in ['MUTAG', 'NCI1', 'PROTEINS', 'COLLAB', 'DD', 'PTC_MR', 'IMDB-BINARY']:
        dataset = get_tudataset_dataset('data/tudataset', data_name)
    else:
        raise NotImplementedError()
    return dataset
