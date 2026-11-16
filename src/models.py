"""Two-layer models compared in the project, each with its original paper's hyperparameters."""
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv

from .gat_layer import GATLayer

# Hyperparameters follow the original papers (GCN: Kipf & Welling 2017; GAT: Velickovic et al. 2018).
CONFIGS = {
    "MLP":   dict(hidden=64, dropout=0.5, lr=0.01,  weight_decay=5e-4),
    "GCN":   dict(hidden=16, dropout=0.5, lr=0.01,  weight_decay=5e-4),
    "SAGE":  dict(hidden=64, dropout=0.5, lr=0.01,  weight_decay=5e-4),
    "GAT":   dict(hidden=8, heads=8, dropout=0.6, lr=0.005, weight_decay=5e-4),
    "GATv2": dict(hidden=8, heads=8, dropout=0.6, lr=0.005, weight_decay=5e-4),
}


class MLP(nn.Module):
    """Ignores the graph entirely: a lower bound for what edges add."""
    uses_graph = False

    def __init__(self, in_dim, n_classes, hidden, dropout, **_):
        super().__init__()
        self.lin1, self.lin2, self.dropout = nn.Linear(in_dim, hidden), nn.Linear(hidden, n_classes), dropout

    def embed(self, x, edge_index=None):
        x = F.dropout(x, self.dropout, self.training)
        return F.relu(self.lin1(x))

    def forward(self, x, edge_index=None):
        return self.lin2(F.dropout(self.embed(x), self.dropout, self.training))


class GCN(nn.Module):
    """Neighbours weighted by 1/sqrt(deg_i deg_j) — fixed by graph structure."""
    uses_graph = True

    def __init__(self, in_dim, n_classes, hidden, dropout, **_):
        super().__init__()
        self.conv1, self.conv2, self.dropout = GCNConv(in_dim, hidden), GCNConv(hidden, n_classes), dropout

    def embed(self, x, edge_index):
        x = F.dropout(x, self.dropout, self.training)
        return F.relu(self.conv1(x, edge_index))

    def forward(self, x, edge_index):
        return self.conv2(F.dropout(self.embed(x, edge_index), self.dropout, self.training), edge_index)


class SAGE(nn.Module):
    """GraphSAGE with mean aggregator: W_self h_i + W_neigh mean_j h_j (full neighbourhood)."""
    uses_graph = True

    def __init__(self, in_dim, n_classes, hidden, dropout, **_):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden, aggr="mean")
        self.conv2 = SAGEConv(hidden, n_classes, aggr="mean")
        self.dropout = dropout

    def embed(self, x, edge_index):
        x = F.dropout(x, self.dropout, self.training)
        return F.relu(self.conv1(x, edge_index))

    def forward(self, x, edge_index):
        return self.conv2(F.dropout(self.embed(x, edge_index), self.dropout, self.training), edge_index)


class GAT(nn.Module):
    """GAT as in the paper: 8 heads x 8 units + ELU, then one output head. `v2=True` swaps in GATv2."""
    uses_graph = True

    def __init__(self, in_dim, n_classes, hidden, heads, dropout, v2=False, **_):
        super().__init__()
        if v2:
            self.conv1 = GATv2Conv(in_dim, hidden, heads=heads, dropout=dropout)
            self.conv2 = GATv2Conv(hidden * heads, n_classes, heads=1, concat=False, dropout=dropout)
        else:
            self.conv1 = GATLayer(in_dim, hidden, heads=heads, dropout=dropout)
            self.conv2 = GATLayer(hidden * heads, n_classes, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def embed(self, x, edge_index):
        x = F.dropout(x, self.dropout, self.training)
        return F.elu(self.conv1(x, edge_index))

    def forward(self, x, edge_index, return_attention=False):
        x = F.dropout(x, self.dropout, self.training)
        h, att1 = self.conv1(x, edge_index, return_attention_weights=True)
        h = F.dropout(F.elu(h), self.dropout, self.training)
        out, att2 = self.conv2(h, edge_index, return_attention_weights=True)
        if return_attention:
            return out, [att1, att2]  # each: (edge_index incl. self-loops, alpha [E, heads])
        return out


def build_model(name, in_dim, n_classes):
    cfg = CONFIGS[name]
    if name == "MLP":
        return MLP(in_dim, n_classes, **cfg)
    if name == "GCN":
        return GCN(in_dim, n_classes, **cfg)
    if name == "SAGE":
        return SAGE(in_dim, n_classes, **cfg)
    if name == "GAT":
        return GAT(in_dim, n_classes, **cfg)
    if name == "GATv2":
        return GAT(in_dim, n_classes, v2=True, **cfg)
    raise ValueError(name)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
