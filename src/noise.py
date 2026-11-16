"""Inject random (noisy) undirected edges into a graph — experiment E3."""
import numpy as np
import torch


def add_random_edges(edge_index, num_nodes, ratio, seed=0):
    """Add `ratio * (#undirected edges)` new random undirected edges.

    New edges join uniformly random node pairs that are not already connected
    (no self-loops, no duplicates). With 7 roughly balanced classes most of them
    are cross-class by chance.

    Returns (new_edge_index, is_noise) where is_noise[k] is True for injected edges.
    Original edges come first, so `new_edge_index[:, :E]` equals the input.
    """
    E = edge_index.size(1)
    n_new = int(round(ratio * (E // 2)))
    if n_new == 0:
        return edge_index.clone(), torch.zeros(E, dtype=torch.bool)

    rng = np.random.default_rng(seed)
    ei = edge_index.cpu().numpy()
    existing = set((ei[0] * num_nodes + ei[1]).tolist())
    new = set()
    while len(new) < n_new:
        k = (n_new - len(new)) * 2
        a = rng.integers(0, num_nodes, k)
        b = rng.integers(0, num_nodes, k)
        for i, j in zip(a.tolist(), b.tolist()):
            if i == j:
                continue
            i, j = min(i, j), max(i, j)
            key = i * num_nodes + j
            if key in existing or key in new:
                continue
            new.add(key)
            if len(new) == n_new:
                break

    new = np.array(sorted(new))
    i, j = new // num_nodes, new % num_nodes
    added = torch.tensor(np.stack([np.concatenate([i, j]), np.concatenate([j, i])]),
                         dtype=edge_index.dtype, device=edge_index.device)
    new_ei = torch.cat([edge_index, added], dim=1)
    is_noise = torch.cat([torch.zeros(E, dtype=torch.bool), torch.ones(added.size(1), dtype=torch.bool)])
    return new_ei, is_noise.to(edge_index.device)


def with_noise(data, ratio, seed=0):
    """Copy of `data` with injected edges; also stores `data.is_noise` for each edge."""
    d = data.clone()
    d.edge_index, d.is_noise = add_random_edges(data.edge_index, data.num_nodes, ratio, seed)
    return d
