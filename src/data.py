"""Dataset loading and simple graph statistics."""
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork

PLANETOID = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}
WEBKB = {"texas": "Texas", "cornell": "Cornell", "wisconsin": "Wisconsin"}
WIKI = {"chameleon": "chameleon", "squirrel": "squirrel"}


def load_dataset(name, root="data"):
    """Return (dataset, data) with row-normalised features.

    Planetoid graphs use the standard public split (Cora: 140/500/1000).
    Low-homophily graphs (Texas, Chameleon, ...) ship 10 splits; pick one with
    `select_split(data, i)`.
    """
    key = name.lower()
    transform = T.NormalizeFeatures()
    if key in PLANETOID:
        ds = Planetoid(root, PLANETOID[key], split="public", transform=transform)
    elif key in WEBKB:
        ds = WebKB(root, WEBKB[key], transform=T.Compose([T.ToUndirected(), transform]))
    elif key in WIKI:
        ds = WikipediaNetwork(root, WIKI[key], geom_gcn_preprocess=True,
                              transform=T.Compose([T.ToUndirected(), transform]))
    else:
        raise ValueError(f"Unknown dataset {name}")
    return ds, ds[0]


def select_split(data, split_idx=0):
    """If masks are 2-D (multiple splits), keep only column `split_idx`."""
    data = data.clone()
    for m in ("train_mask", "val_mask", "test_mask"):
        mask = getattr(data, m)
        if mask.dim() == 2:
            setattr(data, m, mask[:, split_idx % mask.size(1)])
    return data


def edge_homophily(edge_index, y):
    """Fraction of edges whose two endpoints share a label (self-loops excluded)."""
    src, dst = edge_index
    keep = src != dst
    return (y[src[keep]] == y[dst[keep]]).float().mean().item()


def describe(name, data):
    """One-row summary used in the notebook's data table."""
    deg = torch.bincount(data.edge_index[1], minlength=data.num_nodes)
    return {
        "dataset": name,
        "nodes": data.num_nodes,
        "edges (undirected)": data.edge_index.size(1) // 2,
        "features": data.num_features,
        "classes": int(data.y.max()) + 1,
        "train/val/test": f"{int(data.train_mask.sum())}/{int(data.val_mask.sum())}/{int(data.test_mask.sum())}",
        "mean degree": round(deg.float().mean().item(), 2),
        "isolated nodes": int((deg == 0).sum()),
        "edge homophily": round(edge_homophily(data.edge_index, data.y), 3),
    }
