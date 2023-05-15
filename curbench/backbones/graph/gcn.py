'''
GCN model for node and graph classification from tutorials of torch_geometric
https://colab.research.google.com/drive/14OvFnAXggxB8vM4e8vSURUp1TaKnovzX?usp=sharing
https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing
'''


import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


class GCNForNode(nn.Module):
    def __init__(self, dataset, hidden_channels=16):
        super(GCNForNode, self).__init__()
        self.num_classes = dataset.num_classes
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


class GCNForGraph(nn.Module):
    def __init__(self, dataset, hidden_channels=64):
        super(GCNForGraph, self).__init__()
        self.num_classes = dataset.num_classes
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, dataset.num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        x = self.fc(x)
        return x
