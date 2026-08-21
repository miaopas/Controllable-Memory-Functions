"""Results storage: one directory per benchmarked model.

``<run_dir>/run.json``     metadata (model, suite, sizes, parameter counts, causality check, ...)
``<run_dir>/results.csv``  one row per trial (memory function x alpha x size x seed)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union

import pandas as pd

COLUMNS = [
    "model", "size", "n_params", "memory_function", "alpha", "alpha_norm", "seed", "T", "suite",
    "train_loss_min", "train_rloss_min", "val_loss_min", "val_rloss_min", "test_loss", "test_rloss",
    "epochs", "steps", "wall_time_s", "status", "rloss_curve",
]

TRIAL_KEY = ("memory_function", "alpha", "size", "seed")


def trial_key(row: Dict[str, Any]) -> Tuple:
    return (row["memory_function"], round(float(row["alpha"]), 8), int(row["size"]), int(row["seed"]))


def write_meta(run_dir: Path, meta: Dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2, default=str))


def read_meta(run_dir: Union[str, Path]) -> Dict[str, Any]:
    path = Path(run_dir) / "run.json"
    return json.loads(path.read_text()) if path.exists() else {}


def append_row(run_dir: Path, row: Dict[str, Any]) -> None:
    path = run_dir / "results.csv"
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)


def completed_keys(run_dir: Path) -> set:
    path = run_dir / "results.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return {trial_key(r) for r in df.to_dict("records")}


def load_runs(run_dirs: Iterable[Union[str, Path]]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Load and concatenate results from several run directories."""
    frames, metas = [], []
    for d in run_dirs:
        d = Path(d)
        path = d / "results.csv"
        if not path.exists():
            raise FileNotFoundError(f"No results.csv in {d}")
        meta = read_meta(d)
        meta["run_dir"] = str(d)
        metas.append(meta)
        frames.append(pd.read_csv(path))
    if not frames:
        raise ValueError("No run directories given")
    df = pd.concat(frames, ignore_index=True)
    return df, metas
