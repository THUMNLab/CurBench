from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


def get_planetoid_dataset(data_name, data_dir='data/planetoid'):
    dataset = Planetoid(root=data_dir, name=data_name, split='full', transform=NormalizeFeatures())
    dataset.__setattr__('name', data_name)
    return dataset
