"""Plain-PyTorch training loop for one trial.

It mirrors the recipe of the paper's ``LitModule`` (``src/modules/lit_module.py``): MSE loss, Adam
(lr 1e-3), cosine learning-rate decay over the epoch budget, and early stopping on the epoch-mean
training loss. Metrics follow the same definitions:

* ``loss``  = MSE
* ``rloss`` = MSE / Var(y)  (relative MSE, per batch, averaged over the epoch)
"""

from __future__ import annotations

import math
import time
from typing import Dict

import torch
from torch import nn

from .data import Split, TaskData
from .suite import TrainConfig


def pick_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _batches(split: Split, batch_size: int, shuffle: bool, generator: torch.Generator):
    n = split.x.shape[0]
    idx = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for i in range(0, n, batch_size):
        j = idx[i:i + batch_size]
        yield split.x[j], split.y[j]


def _relative_mse(y: torch.Tensor, yhat: torch.Tensor) -> torch.Tensor:
    return torch.mean((y - yhat) ** 2) / torch.var(y)


@torch.no_grad()
def evaluate(model: nn.Module, split: Split, batch_size: int, device: torch.device) -> Dict[str, float]:
    model.eval()
    se, n, rl, nb = 0.0, 0, 0.0, 0
    g = torch.Generator()
    for x, y in _batches(split, batch_size, shuffle=False, generator=g):
        x, y = x.to(device), y.to(device)
        yhat = model(x)
        se += torch.sum((yhat - y) ** 2).item()
        n += y.numel()
        rl += _relative_mse(y, yhat).item()
        nb += 1
    return {"loss": se / max(n, 1), "rloss": rl / max(nb, 1)}


def train_trial(model: nn.Module, task: TaskData, cfg: TrainConfig, seed: int,
                device: torch.device) -> Dict[str, float]:
    """Train ``model`` on ``task`` and return summary metrics."""
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
             if cfg.scheduler == "cosine" else None)
    loss_fn = nn.MSELoss()
    g = torch.Generator().manual_seed(seed)

    best = {"train_loss_min": math.inf, "train_rloss_min": math.inf,
            "val_loss_min": math.inf, "val_rloss_min": math.inf}
    history = []
    bad_epochs, best_monitor = 0, math.inf
    step, epochs_run, status = 0, 0, "ok"
    t_start = time.perf_counter()

    for epoch in range(cfg.epochs):
        model.train()
        tot, rtot, nb = 0.0, 0.0, 0
        for x, y in _batches(task.train, cfg.batch_size, shuffle=True, generator=g):
            x, y = x.to(device), y.to(device)
            yhat = model(x)
            loss = loss_fn(yhat, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tot += loss.item()
            rtot += _relative_mse(y, yhat.detach()).item()
            nb += 1
            step += 1
        if sched is not None:
            sched.step()
        epochs_run = epoch + 1

        train_loss, train_rloss = tot / nb, rtot / nb
        if not math.isfinite(train_loss):
            status = "diverged"
            break
        val = evaluate(model, task.val, cfg.batch_size, device)
        best["train_loss_min"] = min(best["train_loss_min"], train_loss)
        best["train_rloss_min"] = min(best["train_rloss_min"], train_rloss)
        best["val_loss_min"] = min(best["val_loss_min"], val["loss"])
        best["val_rloss_min"] = min(best["val_rloss_min"], val["rloss"])
        history.append(train_rloss)

        # early stopping on the training loss (paper: EarlyStopping(monitor="train/loss_epoch"))
        if train_loss < best_monitor:
            best_monitor, bad_epochs = train_loss, 0
        else:
            bad_epochs += 1
        if epochs_run >= cfg.min_epochs and cfg.patience and bad_epochs >= cfg.patience:
            break

    test = evaluate(model, task.test, cfg.batch_size, device) if status == "ok" else {
        "loss": math.nan, "rloss": math.nan}
    return {
        **best,
        "test_loss": test["loss"],
        "test_rloss": test["rloss"],
        "epochs": epochs_run,
        "steps": step,
        "wall_time_s": time.perf_counter() - t_start,
        "status": status,
        "rloss_curve": " ".join(f"{v:.4g}" for v in history),
    }
