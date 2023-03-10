from .planetoid import get_planetoid_dataset
from .tudataset import get_tudataset_dataset, split_dataset
from .utils import LabelNoise

supported_datasets = [
    'mutag', 'nci1', 'proteins', 'collab', 'dd', 'ptc_mr', 'imdb-binary'
]

task_graph_label_range_map = {
    'mutag': [0, 1],
    'nci1': [0, 1],
    'proteins': [0, 1],
    'collab': [0, 2],
    'dd': [0, 1], 
    'ptc_mr': [0, 1],
    'imdb-binary': [0, 1],
}

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
    
    assert data_name in supported_datasets, \
        'Assert Error: data_name should be in ' + str(supported_datasets)

    dataset = get_dataset(data_name)
    train_dataset, valid_dataset, test_dataset = split_dataset(dataset)
    if noise_ratio > 0.0:
        label_range = task_graph_label_range_map[data_name]
        train_dataset = LabelNoise(train_dataset, noise_ratio, label_range)
    return dataset, train_dataset, valid_dataset, test_dataset
