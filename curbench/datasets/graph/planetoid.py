from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


def get_planetoid_dataset(data_dir, data_name):
    return Planetoid(root=data_dir, name=data_name, split='full', transform=NormalizeFeatures())
