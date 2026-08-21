"""Dataset generation for one (memory function, alpha, T, seed) setting.

This is a thin wrapper around the paper's generator (``src/data/lf/generator.py``), so targets are
exactly the ones used in the paper experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

import torch

from src.data.lf.generator import LF

from .suite import DataConfig


@dataclass
class Split:
    x: torch.Tensor  # (N, T, 1)
    y: torch.Tensor  # (N, T, 1)


@dataclass
class TaskData:
    train: Split
    val: Split
    test: Split
    rho: torch.Tensor  # (T,) memory function values rho(0..T-1)


@lru_cache(maxsize=8)
def _generate(function: str, alpha: float, T: int, seed: int, data: DataConfig,
              params: Tuple[Tuple[str, object], ...]) -> TaskData:
    n = data.n_train + data.n_val + data.n_test
    lf = LF(
        data_dir="data/lf",
        N=n,
        T=T,
        D=1,
        rho_name=function,
        rho_param={"alpha": alpha, **dict(params)},
        inner_activation=data.inner_activation,
        output_activation=data.output_activation,
        input_generator=data.input_generator,
        save_file=False,
        manual_seed=seed,
    )
    dataset, rho_vals = lf.dataset_construction()
    x, y = dataset.tensors
    a, b = data.n_train, data.n_train + data.n_val
    return TaskData(
        train=Split(x[:a], y[:a]),
        val=Split(x[a:b], y[a:b]),
        test=Split(x[b:], y[b:]),
        rho=rho_vals[0].float(),
    )


def make_task(function: str, alpha: float, T: int, seed: int, data: DataConfig,
              params: dict | None = None) -> TaskData:
    """Generate (and cache) the train/val/test data for one setting."""
    frozen = tuple(sorted((params or {}).items()))
    return _generate(function, float(alpha), int(T), int(seed), data, frozen)
