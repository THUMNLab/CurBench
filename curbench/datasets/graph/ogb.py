from ogb.graphproppred import PygGraphPropPredDataset


def get_ogb_dataset(data_name, data_dir='data/ogb'):
    if data_name[3] == 'g':     # 'ogbg-'
        dataset = PygGraphPropPredDataset(name=data_name, root=data_dir)
    else:                       # 'ogbn-' or 'ogbl-'
        raise NotImplementedError()
    dataset.__setattr__('name', data_name)

    split_idx = dataset.get_idx_split()

    return dataset, dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]