"""Results storage: one directory per benchmarked model.

``<run_dir>/run.json``     metadata (model, suite, sizes, parameter counts, causality check, ...)
``<run_dir>/results.csv``  one row per trial (memory function x alpha x size x seed)
``<run_dir>/results.shard-I-of-N.csv``  the same, written by shard I of an N-way array job
                                       (see ``--shard``); all result files are read together.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd

COLUMNS = [
    "model", "size", "n_params", "memory_function", "alpha", "alpha_norm", "seed", "T", "suite",
    "train_loss_min", "train_rloss_min", "val_loss_min", "val_rloss_min", "test_loss", "test_rloss",
    "epochs", "steps", "wall_time_s", "status", "rloss_curve",
]

TRIAL_KEY = ("memory_function", "alpha", "size", "seed")


def trial_key(row: Dict[str, Any]) -> Tuple:
    return (row["memory_function"], round(float(row["alpha"]), 8), int(row["size"]), int(row["seed"]))


def results_file(shard: Optional[Tuple[int, int]] = None) -> str:
    return "results.csv" if shard is None else f"results.shard-{shard[0]}-of-{shard[1]}.csv"


def result_files(run_dir: Union[str, Path]) -> List[Path]:
    run_dir = Path(run_dir)
    return sorted(p for p in [run_dir / "results.csv", *run_dir.glob("results.shard-*.csv")]
                  if p.exists())


def write_meta(run_dir: Path, meta: Dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    # write-then-rename, so concurrent shards never leave a half-written file
    tmp = run_dir / f".run.json.{os.getpid()}"
    tmp.write_text(json.dumps(meta, indent=2, default=str))
    os.replace(tmp, run_dir / "run.json")


def read_meta(run_dir: Union[str, Path]) -> Dict[str, Any]:
    path = Path(run_dir) / "run.json"
    return json.loads(path.read_text()) if path.exists() else {}


def append_row(run_dir: Path, row: Dict[str, Any], filename: str = "results.csv") -> None:
    path = run_dir / filename
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)


def _read_results(run_dir: Union[str, Path]) -> pd.DataFrame:
    files = result_files(run_dir)
    if not files:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    # a trial can appear twice if two shards overlapped after a restart; keep the first
    keys = df.apply(lambda r: trial_key(r), axis=1)
    return df[~keys.duplicated()].reset_index(drop=True)


def completed_keys(run_dir: Path) -> set:
    return {trial_key(r) for r in _read_results(run_dir).to_dict("records")}


def load_runs(run_dirs: Iterable[Union[str, Path]]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Load and concatenate results from several run directories."""
    frames, metas = [], []
    for d in run_dirs:
        d = Path(d)
        if not result_files(d):
            raise FileNotFoundError(f"No results.csv in {d}")
        meta = read_meta(d)
        meta.setdefault("run_dir", str(d))
        metas.append(meta)
        frames.append(_read_results(d))
    if not frames:
        raise ValueError("No run directories given")
    df = pd.concat(frames, ignore_index=True)
    return df, metas
