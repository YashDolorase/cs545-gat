"""Quick checks: `python -m pytest tests -q` (or `python tests/test_sanity.py`)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.attention import get_attention, node_attention_stats  # noqa: E402
from src.gat_layer import check_against_pyg, segment_softmax  # noqa: E402
from src.noise import add_random_edges  # noqa: E402


def toy_graph():
    torch.manual_seed(0)
    ei = torch.tensor([[0, 1, 1, 2, 2, 3, 0, 3, 3, 4], [1, 0, 2, 1, 3, 2, 3, 0, 4, 3]])
    return torch.randn(5, 6), ei


def test_segment_softmax_sums_to_one():
    scores = torch.randn(10, 3)
    index = torch.tensor([0, 0, 1, 1, 1, 2, 3, 3, 3, 3])
    a = segment_softmax(scores, index, 4)
    sums = torch.zeros(4, 3).index_add(0, index, a)
    assert torch.allclose(sums, torch.ones(4, 3), atol=1e-6)


def test_gat_layer_matches_pyg():
    x, ei = toy_graph()
    for concat in (True, False):
        r = check_against_pyg(x, ei, 6, 4, heads=3, concat=concat)
        assert r["match"], r


def test_noise_edges():
    _, ei = toy_graph()
    new, is_noise = add_random_edges(ei, 5, ratio=0.6, seed=1)  # 5 undirected edges -> add 3
    assert is_noise.sum() == 6 and torch.equal(new[:, :ei.size(1)], ei)
    keys = (new[0] * 5 + new[1]).tolist()
    assert len(keys) == len(set(keys)) and (new[0] != new[1]).all()
    rev = set(zip(new[1].tolist(), new[0].tolist()))
    assert set(zip(new[0].tolist(), new[1].tolist())) == rev  # undirected


def test_attention_stats_uniform():
    from src.models import GAT
    x, ei = toy_graph()
    model = GAT(6, 2, hidden=4, heads=2, dropout=0.0)
    torch.nn.init.zeros_(model.conv1.att_src)
    torch.nn.init.zeros_(model.conv1.att_dst)

    class D:  # minimal data stub
        pass
    d = D()
    d.x, d.edge_index = x, ei
    e, a = get_attention(model, d)[0]
    stats = node_attention_stats(e, a, torch.tensor([0, 0, 1, 1, 0]), 5)
    assert torch.allclose(torch.tensor(stats.entropy.values), torch.tensor(stats.uniform_entropy.values), atol=1e-5)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
