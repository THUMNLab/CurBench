import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATConv, GlobalAttention, Set2Set, \
                               global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import degree, add_self_loops, remove_self_loops, softmax
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


### GIN convolution along the graph structure
class GINConvOGB(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConvOGB, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


### GAT convolution along the graph structure
class GATConvOGB(GATConv):
    def __init__(self, in_dim, out_dim):
        super().__init__(in_dim,out_dim, add_self_loops = False)
        self.bond_encoder = BondEncoder(emb_dim = out_dim)

    def forward(self, x, edge_index, edge_attr, 
                size = None, return_attention_weights=None):
        edge_embedding = self.bond_encoder(edge_attr)
        H, C = self.heads, self.out_channels

        x_src = None
        x_dst = None
        alpha_src = None
        alpha_dst = None
        if isinstance(x, torch.Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_src = x_dst = self.lin_src(x).view(-1, H, C)
            alpha_src = (x_src * self.att_src).sum(dim=-1)
            alpha_dst = (x_dst * self.att_dst).sum(dim=-1)
        else:
            x_src, x_dst = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_src = self.lin_src(x_src).view(-1, H, C)
            alpha_src = (x_src * self.att_src).sum(dim=-1)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)
                alpha_dst = (x_dst * self.att_dst).sum(dim=-1)

        assert x_src is not None
        assert alpha_src is not None

        if self.add_self_loops:
            if isinstance(edge_index, torch.Tensor):
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            # elif isinstance(edge_index, SparseTensor):
            #     edge_index = set_diag(edge_index)

        out = self.propagate(edge_index, x=(x_src, x_dst),
                             alpha=(alpha_src, alpha_dst), size=size, edge_attr = edge_embedding)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, torch.Tensor):
                return out, (edge_index, alpha)
            # elif isinstance(edge_index, SparseTensor):
            #     return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def main_para(self):
        return [self.lin_src.weight]

    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i, edge_attr):
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return F.relu(x_j + edge_attr.unsqueeze(1)) * alpha.unsqueeze(-1)


### GCN convolution along the graph structure
class GCNConvOGB(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConvOGB, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


### GNN to generate node embedding
class GNNNodeOGB(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin'):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers

        '''

        super(GNNNodeOGB, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConvOGB(emb_dim))
            elif gnn_type == 'gat':
                self.convs.append(GATConvOGB(emb_dim, emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConvOGB(emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        ### computing input node embedding

        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation


class GNNForOGB(torch.nn.Module):

    def __init__(self, num_classes, num_layer = 5, emb_dim = 300, 
                    gnn_type = 'gin', virtual_node = False, residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean"):
        '''
            num_classes (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNNForOGB, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        self.gnn_node = GNNNodeOGB(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_classes)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_classes)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        h_graph = self.pool(h_node, batched_data.batch)

        return self.graph_pred_linear(h_graph)
