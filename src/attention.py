"""Tools for reading what a trained GAT pays attention to (E2, E3)."""
import numpy as np
import pandas as pd
import torch


@torch.no_grad()
def get_attention(model, data):
    """Return per-layer list of (edge_index incl. self-loops, alpha averaged over heads [E])."""
    model.eval()
    _, atts = model(data.x, data.edge_index, return_attention=True)
    return [(ei.cpu(), a.mean(dim=1).cpu()) for ei, a in atts]


def node_attention_stats(edge_index, alpha, y, num_nodes):
    """Per-node statistics of one layer's attention.

    entropy          : -sum_j alpha_ij log alpha_ij over N(i) ∪ {i}
    uniform_entropy  : log(deg_i + 1), entropy if every neighbour got the same weight
    self_att         : weight on the node itself
    same_class_att   : share of *neighbour* attention (self excluded, renormalised) on same-class neighbours
    same_class_frac  : share of neighbours with the same class = what uniform weights would give
    """
    y = y.cpu()
    src, dst = edge_index
    is_self = src == dst
    ent_terms = -(alpha * torch.log(alpha.clamp_min(1e-12)))
    entropy = torch.zeros(num_nodes).index_add(0, dst, ent_terms)
    deg = torch.bincount(dst[~is_self], minlength=num_nodes).float()
    self_att = torch.zeros(num_nodes).index_add(0, dst[is_self], alpha[is_self])

    ns, nd, na = src[~is_self], dst[~is_self], alpha[~is_self]
    same = (y[ns] == y[nd]).float()
    neigh_att = torch.zeros(num_nodes).index_add(0, nd, na)
    same_att = torch.zeros(num_nodes).index_add(0, nd, na * same)
    same_cnt = torch.zeros(num_nodes).index_add(0, nd, same)

    df = pd.DataFrame({
        "degree": deg.numpy(),
        "entropy": entropy.numpy(),
        "uniform_entropy": torch.log(deg + 1).numpy(),
        "self_att": self_att.numpy(),
        "same_class_att": (same_att / neigh_att.clamp_min(1e-12)).numpy(),
        "same_class_frac": (same_cnt / deg.clamp_min(1)).numpy(),
        "label": y.numpy(),
    })
    df.loc[df.degree == 0, ["same_class_att", "same_class_frac"]] = np.nan
    return df


def normalized_edge_attention(edge_index, alpha, num_nodes):
    """alpha_ij * (deg_i + 1): 1.0 means 'exactly uniform', >1 above average, <1 below."""
    dst = edge_index[1]
    size = torch.bincount(dst, minlength=num_nodes).float()  # deg + self-loop
    return alpha * size[dst]


def edge_mask_for(edge_index, ref_edge_index, ref_mask, num_nodes):
    """Map a boolean mask defined on `ref_edge_index` onto the (reordered) `edge_index`."""
    keys = edge_index[0].long() * num_nodes + edge_index[1].long()
    ref_keys = (ref_edge_index[0].long() * num_nodes + ref_edge_index[1].long()).cpu()
    return torch.isin(keys, ref_keys[ref_mask.cpu()])


def noise_attention_summary(model, data):
    """Mean normalised attention on original vs injected edges, per layer (self-loops excluded)."""
    rows = []
    for layer, (ei, a) in enumerate(get_attention(model, data), start=1):
        norm = normalized_edge_attention(ei, a, data.num_nodes)
        noise = edge_mask_for(ei, data.edge_index, data.is_noise, data.num_nodes)
        not_self = ei[0] != ei[1]
        rows.append({
            "layer": layer,
            "att_original": norm[not_self & ~noise].mean().item(),
            "att_injected": norm[not_self & noise].mean().item() if (noise.any()) else float("nan"),
        })
    return rows
