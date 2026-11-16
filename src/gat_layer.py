"""A Graph Attention layer written from scratch (Velickovic et al., 2018).

For every edge j -> i and every head k:

    e_ij   = LeakyReLU( a_src^k . W^k h_j  +  a_dst^k . W^k h_i )     (= a^T [W h_i || W h_j])
    alpha_ij = softmax over j in N(i) ∪ {i} of e_ij
    h_i'   = sum_j alpha_ij W^k h_j

Heads are concatenated (hidden layers) or averaged (output layer).

Only plain PyTorch ops are used; `check_against_pyg` copies the weights into
PyG's GATConv and confirms both layers produce identical outputs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def segment_softmax(scores, index, num_nodes):
    """Softmax of `scores` [E, H] within groups of edges sharing the same target `index` [E]."""
    idx = index.unsqueeze(-1).expand_as(scores)
    # subtract per-group max for numerical stability
    group_max = torch.full((num_nodes, scores.size(1)), float("-inf"), device=scores.device, dtype=scores.dtype)
    group_max = group_max.scatter_reduce(0, idx, scores, reduce="amax", include_self=True)
    exp = (scores - group_max[index]).exp()
    denom = torch.zeros((num_nodes, scores.size(1)), device=scores.device, dtype=scores.dtype)
    denom = denom.index_add(0, index, exp)
    return exp / (denom[index] + 1e-16)


def add_self_loops(edge_index, num_nodes):
    """Drop existing self-loops, then append one self-loop per node (same as GATConv)."""
    src, dst = edge_index
    keep = src != dst
    loops = torch.arange(num_nodes, device=edge_index.device)
    return torch.cat([edge_index[:, keep], torch.stack([loops, loops])], dim=1)


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=1, concat=True, negative_slope=0.2,
                 dropout=0.0, self_loops=True, bias=True):
        super().__init__()
        self.in_dim, self.out_dim, self.heads, self.concat = in_dim, out_dim, heads, concat
        self.negative_slope, self.dropout, self.self_loops = negative_slope, dropout, self_loops

        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, out_dim))  # scores the neighbour j
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_dim))  # scores the centre node i
        self.bias = nn.Parameter(torch.zeros(heads * out_dim if concat else out_dim)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, return_attention_weights=False):
        N, H, C = x.size(0), self.heads, self.out_dim
        if self.self_loops:
            edge_index = add_self_loops(edge_index, N)
        src, dst = edge_index  # message flows src (j) -> dst (i)

        h = self.W(x).view(N, H, C)                          # W h            [N, H, C]
        score_src = (h * self.att_src).sum(-1)               # a_src . W h_j  [N, H]
        score_dst = (h * self.att_dst).sum(-1)               # a_dst . W h_i  [N, H]
        e = F.leaky_relu(score_src[src] + score_dst[dst], self.negative_slope)  # [E, H]

        alpha = segment_softmax(e, dst, N)                   # normalise over each neighbourhood
        alpha_drop = F.dropout(alpha, p=self.dropout, training=self.training)

        out = torch.zeros(N, H, C, device=x.device, dtype=h.dtype)
        out = out.index_add(0, dst, h[src] * alpha_drop.unsqueeze(-1))
        out = out.reshape(N, H * C) if self.concat else out.mean(dim=1)
        if self.bias is not None:
            out = out + self.bias

        if return_attention_weights:
            return out, (edge_index, alpha)
        return out

    def extra_repr(self):
        return f"{self.in_dim}, {self.out_dim}, heads={self.heads}, concat={self.concat}"


@torch.no_grad()
def copy_to_gatconv(layer):
    """Build a PyG GATConv holding exactly the same weights as `layer`."""
    from torch_geometric.nn import GATConv
    conv = GATConv(layer.in_dim, layer.out_dim, heads=layer.heads, concat=layer.concat,
                   negative_slope=layer.negative_slope, dropout=layer.dropout,
                   add_self_loops=layer.self_loops, bias=layer.bias is not None)
    lin = conv.lin if getattr(conv, "lin", None) is not None else conv.lin_src  # PyG >=2.4 vs older
    lin.weight.copy_(layer.W.weight)
    conv.att_src.copy_(layer.att_src)
    conv.att_dst.copy_(layer.att_dst)
    if layer.bias is not None:
        conv.bias.copy_(layer.bias)
    return conv


@torch.no_grad()
def check_against_pyg(x, edge_index, in_dim, out_dim, heads=8, concat=True, seed=0, atol=1e-5):
    """Return max |difference| between our layer and GATConv (outputs and attention)."""
    torch.manual_seed(seed)
    ours = GATLayer(in_dim, out_dim, heads=heads, concat=concat).eval()
    pyg = copy_to_gatconv(ours).eval()
    out_a, (ei_a, att_a) = ours(x, edge_index, return_attention_weights=True)
    out_b, (ei_b, att_b) = pyg(x, edge_index, return_attention_weights=True)

    # GATConv may order edges differently; compare attention per (src, dst) pair
    N = x.size(0)
    key_a, key_b = ei_a[0] * N + ei_a[1], ei_b[0] * N + ei_b[1]
    att_a = att_a[key_a.argsort()]
    att_b = att_b[key_b.argsort()]
    out_diff = (out_a - out_b).abs().max().item()
    att_diff = (att_a - att_b).abs().max().item()
    return {"output_max_abs_diff": out_diff, "attention_max_abs_diff": att_diff,
            "match": out_diff < atol and att_diff < atol}
