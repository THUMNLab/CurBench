import torch.nn as nn
from torch_geometric.nn import MLP, GINConv, global_add_pool


class GIN(nn.Module):
    def __init__(self, dataset, hidden_channels=32, num_layers=5, dropout=0.5):
        super().__init__()
        self.num_classes = dataset.num_classes
        self.convs = nn.ModuleList()
        in_channels = dataset.num_features
        for _ in range(num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels
        self.mlp = MLP([hidden_channels, hidden_channels, dataset.num_classes],
                       norm=None, dropout=dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_add_pool(x, batch)
        return self.mlp(x)
