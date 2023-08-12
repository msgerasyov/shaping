import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.utils import sort_edge_index


class GCN(torch.nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden_dim: int = 16, dropout: float = 0.0):
        super().__init__()
        self.conv1 = GCNConv(n_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, n_classes)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class GAT(torch.nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden_dim: int = 16, dropout: float = 0.0, n_heads: int = 4):
        super().__init__()
        self.conv1 = GATConv(n_features, hidden_dim, heads=n_heads)
        self.conv2 = GATConv(hidden_dim * n_heads, n_classes, heads=1)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class GraphSAGE(torch.nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden_dim: int = 16, dropout: float = 0.0):
        super().__init__()
        self.conv1 = SAGEConv(n_features, hidden_dim, aggr="mean")
        self.conv2 = SAGEConv(hidden_dim, n_classes, aggr="lstm")
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, sort_edge_index(data.edge_index, sort_by_row=False)

        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
