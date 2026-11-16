# Graph Attention Networks: when does learned neighbour weighting help?

CS545 (Fall 2026) individual project — Hansi Yasodara.

We explain Graph Attention Networks (GAT; Veličković et al., 2018), implement the GAT layer from scratch
(verified against PyG's `GATConv`), and compare it with an MLP, GCN and GraphSAGE on Cora and CiteSeer.

| Experiment | Question |
|---|---|
| **E1 Benchmark** | Does GAT beat GCN / GraphSAGE by more than the spread over 10 random seeds? |
| **E2 Attention** | Are learned attention weights close to uniform, or concentrated on same-class neighbours? |
| **E3 Noisy neighbours** | With 0–100% extra random edges, does GAT degrade more slowly and attend less to injected edges? |
| Stretch | GATv2; low-homophily graphs (Texas, Chameleon) |

## Layout
```
GAT_Project.ipynb                    report notebook (the deliverable)
notebooks/01_run_experiments_colab.ipynb   runs every experiment on a Colab GPU
src/gat_layer.py    GAT layer from scratch + equivalence check with GATConv
src/models.py       MLP, GCN, GraphSAGE, GAT, GATv2 (paper hyperparameters)
src/train.py        training loop, early stopping on validation accuracy
src/noise.py        random edge injection (E3)
src/attention.py    attention entropy, same-class share, injected-vs-original attention
scripts/run_experiments.py   E1 / E3 sweeps -> results/*.csv
tests/test_sanity.py
results/            CSVs + one trained GAT per dataset (small)
```

## Reproduce
Open `notebooks/01_run_experiments_colab.ipynb` in Colab (GPU runtime) and run all, or locally:
```bash
pip install -r requirements.txt
python -m pytest tests -q
python scripts/run_experiments.py --exp e1      # ~10 min on a T4
python scripts/run_experiments.py --exp e3
```
