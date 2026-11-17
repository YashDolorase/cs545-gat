"""Run the project's experiments and append results to CSV files in results/.

    python scripts/run_experiments.py --exp e1                # benchmark, Cora + CiteSeer, 10 seeds
    python scripts/run_experiments.py --exp e3                # noisy edges on Cora
    python scripts/run_experiments.py --exp e1 --models GAT GATv2 --datasets Texas Chameleon   # stretch

Runs already present in the CSV are skipped, so an interrupted Colab session can simply be re-run.
"""
import argparse
import os
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.attention import noise_attention_summary  # noqa: E402
from src.data import load_dataset, select_split  # noqa: E402
from src.noise import with_noise  # noqa: E402
from src.train import get_device, train_model  # noqa: E402

ALL_MODELS = ["MLP", "GCN", "SAGE", "GAT"]


def done_keys(path, cols):
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    return set(map(tuple, df[cols].astype(str).values.tolist()))


def append_row(path, row):
    """Rewrite the whole CSV so rows with extra columns (GAT attention stats) keep a valid header."""
    df = pd.DataFrame([row])
    if os.path.exists(path):
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)


def run_e1(args, device):
    path = os.path.join(args.out, "e1_benchmark.csv")
    done = done_keys(path, ["dataset", "model", "seed"])
    for ds_name in args.datasets or ["Cora", "CiteSeer"]:
        _, data = load_dataset(ds_name, args.data_root)
        for seed in range(args.seeds):
            d = select_split(data, seed)  # no-op for Planetoid's single public split
            for name in args.models or ALL_MODELS:
                if (ds_name, name, str(seed)) in done:
                    continue
                model, m = train_model(name, d, seed=seed, device=device)
                append_row(path, {"dataset": ds_name, **m})
                print(f"[E1] {ds_name:9s} {name:6s} seed {seed}: test {m['test_acc']:.3f} "
                      f"(best ep {m['best_epoch']}, {m['ms_per_epoch']:.1f} ms/ep)", flush=True)
                if name in ("GAT", "GATv2") and seed == 0:  # keep one trained GAT for E2 plots
                    torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()},  # fp16 keeps files small
                               os.path.join(args.out, f"{name}_{ds_name}_seed0.pt"))


def run_e3(args, device):
    path = os.path.join(args.out, "e3_noise.csv")
    done = done_keys(path, ["dataset", "noise", "model", "seed"])
    for ds_name in args.datasets or ["Cora"]:
        _, data = load_dataset(ds_name, args.data_root)
        for ratio in args.noise:
            for seed in range(args.seeds):
                noisy = with_noise(select_split(data, seed), ratio, seed=seed)  # same noisy graph for every model
                y = noisy.y
                src, dst = noisy.edge_index[:, noisy.is_noise]
                cross = (y[src] != y[dst]).float().mean().item() if src.numel() else float("nan")
                for name in args.models or ALL_MODELS:
                    if (ds_name, str(ratio), name, str(seed)) in done:
                        continue
                    model, m = train_model(name, noisy, seed=seed, device=device)
                    row = {"dataset": ds_name, "noise": ratio, "injected_cross_class": cross, **m}
                    if name in ("GAT", "GATv2"):
                        for r in noise_attention_summary(model, noisy.to(device)):
                            row[f"L{r['layer']}_att_original"] = r["att_original"]
                            row[f"L{r['layer']}_att_injected"] = r["att_injected"]
                    append_row(path, row)
                    print(f"[E3] {ds_name} noise {ratio:.2f} {name:6s} seed {seed}: test {m['test_acc']:.3f}",
                          flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", choices=["e1", "e3"], required=True)
    p.add_argument("--datasets", nargs="+")
    p.add_argument("--models", nargs="+")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5, 1.0])
    p.add_argument("--out", default="results")
    p.add_argument("--data_root", default="data")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = get_device()
    print("device:", device, "| torch", torch.__version__)
    t0 = time.time()
    {"e1": run_e1, "e3": run_e3}[args.exp](args, device)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
