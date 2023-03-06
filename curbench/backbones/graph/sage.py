import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool


class SageForNode(nn.Module):
    def __init__(self, dataset, hidden_channels=16):
        super(SageForNode, self).__init__()
        self.num_labels = dataset.num_classes
        self.sage1 = SAGEConv(dataset.num_features, hidden_channels)
        self.sage2 = SAGEConv(hidden_channels, dataset.num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.sage1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.sage2(x, edge_index)
        return x


class SageForGraph(nn.Module):
    def __init__(self, dataset, hidden_channels=64):
        super(SageForGraph, self).__init__()
        self.num_labels = dataset.num_classes
        self.sage1 = SAGEConv(dataset.num_features, hidden_channels)
        self.sage2 = SAGEConv(hidden_channels, hidden_channels)
        self.sage3 = SAGEConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, dataset.num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.sage1(x, edge_index)
        x = self.relu(x)
        x = self.sage2(x, edge_index)
        x = self.relu(x)
        x = self.sage3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        x = self.fc(x)
        return x
