"""Full-batch training with early stopping on validation accuracy."""
import copy
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .models import CONFIGS, build_model, count_params


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    logits = model(data.x, data.edge_index)
    pred = logits.argmax(-1)
    res = {}
    for split in ("train", "val", "test"):
        mask = getattr(data, f"{split}_mask")
        res[f"{split}_acc"] = (pred[mask] == data.y[mask]).float().mean().item()
        res[f"{split}_loss"] = F.cross_entropy(logits[mask], data.y[mask]).item()
    return res


def train_model(name, data, seed=0, max_epochs=1000, patience=100, device=None, verbose=False):
    """Train one model; keep the weights with best val accuracy (ties -> lower val loss).

    Returns (model with best weights loaded, metrics dict).
    """
    device = device or get_device()
    data = data.to(device)
    set_seed(seed)
    n_classes = int(data.y.max()) + 1
    model = build_model(name, data.num_features, n_classes).to(device)
    cfg = CONFIGS[name]
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    best = (-1.0, float("inf"))
    best_state, best_epoch, bad, epoch_times = None, 0, 0, []
    for epoch in range(1, max_epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_times.append(time.perf_counter() - t0)

        m = evaluate(model, data)
        score = (m["val_acc"], m["val_loss"])
        if score[0] > best[0] or (score[0] == best[0] and score[1] < best[1]):
            best, best_epoch, bad = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if verbose and epoch % 50 == 0:
            print(f"{name} ep {epoch:4d} loss {loss.item():.4f} val {m['val_acc']:.3f}")
        if bad >= patience:
            break

    model.load_state_dict(best_state)
    final = evaluate(model, data)
    metrics = {
        "model": name, "seed": seed,
        "test_acc": final["test_acc"], "val_acc": final["val_acc"], "train_acc": final["train_acc"],
        "best_epoch": best_epoch, "epochs_run": epoch,
        "ms_per_epoch": 1000 * float(np.mean(epoch_times[1:] or epoch_times)),  # skip warm-up epoch
        "params": count_params(model),
    }
    return model, metrics
