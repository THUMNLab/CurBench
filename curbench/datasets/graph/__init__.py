from .planetoid import get_planetoid_dataset
from .tudataset import get_tudataset_dataset
from .ogb import get_ogb_dataset
from .utils import LabelNoise, ClassImbalanced
from torchmetrics.functional.classification import accuracy, auroc


name_trans = {
    'cora':     'Cora', 
    'citeseer': 'CiteSeer', 
    'pubmed':   'PubMed',

    'mutag':    'MUTAG', 
    'nci1':     'NCI1', 
    'proteins': 'PROTEINS', 
    'dd':       'DD', 
    'ptc_mr':   'PTC_MR',

    'molhiv':   'ogbg-molhiv',
}

data_dict = {
    'cora':     get_planetoid_dataset, 
    'citeseer': get_planetoid_dataset,
    'pubmed':   get_planetoid_dataset,

    'mutag':    get_tudataset_dataset,
    'nci1':     get_tudataset_dataset,
    'proteins': get_tudataset_dataset,
    'dd':       get_tudataset_dataset,
    'ptc_mr':   get_tudataset_dataset,

    'molhiv':   get_ogb_dataset,
}

task_graph_label_range_map = {
    'mutag':    [0, 1],
    'nci1':     [0, 1],
    'proteins': [0, 1],
    'dd':       [0, 1], 
    'ptc_mr':   [0, 1],
    'molhiv':   [0, 1],
}

def get_dataset(data_name):
    assert not ('noise' in data_name and 'imbalance' in data_name), \
        'Assert Error: only support one setting from [standard, noise, imbalance]'
    
    # noise setting
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
        noise_ratio = None

    # imbalance setting
    if 'imbalance' in data_name:
        try:
            parts = data_name.split('-')
            data_name = parts[0]
            imbalance_ratio = int(parts[-1])
        except:
            assert False, 'Assert Error: data_name shoule be [dataset]-imbalance-[ratio]'
        assert imbalance_ratio >= 1 and imbalance_ratio <= 200, \
            'Assert Error: imbalance ratio should be in range of [1, 200]'
    else:
        imbalance_ratio = None

    # get standard, noisy or imbalanced dataset
    assert data_name in data_dict, 'Assert Error: data_name should be in ' + str(list(data_dict.keys()))
    # dataset, train_idx, valid_idx, test_idx = data_dict[data_name](name_trans[data_name])
    if data_name in ['cora', 'citeseer', 'pubmed']:
        raise NotImplementedError()
    else:
        dataset, train_dataset, valid_dataset, test_dataset = data_dict[data_name](name_trans[data_name])
        if noise_ratio: 
            label_range = task_graph_label_range_map[data_name]
            train_dataset = LabelNoise(train_dataset, noise_ratio, label_range)
        if imbalance_ratio: 
            train_dataset = ClassImbalanced(train_dataset, imbalance_ratio) 
        return dataset, train_dataset, valid_dataset, test_dataset


def get_metric(data_name):
    # allow data name format: [data]-[noise/imbalance]-[args]
    data_name = data_name.split('-')[0]
    assert data_name in data_dict, \
            'Assert Error: data_name should be in ' + str(list(data_dict.keys()))
    if data_name == 'molhiv':
        return auroc, 'ROC AUC'
    else:
        return accuracy, 'Acc'