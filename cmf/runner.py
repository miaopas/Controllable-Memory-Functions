"""Local runner: plan the trial grid, train each trial in-process, write results incrementally.

Re-running with the same output directory resumes: finished trials are skipped.
"""

from __future__ import annotations

import datetime as _dt
import math
import multiprocessing
import pickle
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

import torch

from . import __version__
from .data import make_task
from .results import append_row, completed_keys, read_meta, trial_key, write_meta
from .spec import ModelSpec, check_model, count_parameters
from .suite import Suite, load_suite
from .training import pick_device, train_trial


def _fmt_time(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "?"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def plan_trials(suite: Suite, sizes: Sequence[int]) -> List[dict]:
    trials = []
    for fn in suite.functions:
        a_max = max(fn.alpha)
        for alpha in fn.alpha:
            for size in sizes:
                for seed in suite.seeds:
                    trials.append({
                        "memory_function": fn.name, "alpha": alpha, "alpha_norm": alpha / a_max,
                        "size": int(size), "seed": int(seed), "T": suite.T_for(fn),
                        "params": fn.params,
                    })
    return trials


def _init_worker() -> None:
    import warnings

    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)


def run_trial(spec: ModelSpec, t: dict, suite: Suite, device: str) -> dict:
    """Train one trial and return its results row (runs in the main or a worker process)."""
    task = make_task(t["memory_function"], t["alpha"], t["T"], t["seed"], suite.data, t["params"])
    torch.manual_seed(t["seed"])
    model = spec.build(t["size"])
    metrics = train_trial(model, task, suite.training, seed=t["seed"], device=torch.device(device))
    return {
        "model": spec.name, "size": t["size"], "n_params": count_parameters(model),
        "memory_function": t["memory_function"], "alpha": t["alpha"],
        "alpha_norm": round(t["alpha_norm"], 6), "seed": t["seed"], "T": t["T"],
        "suite": suite.name, **metrics,
    }


def run_benchmark(
    spec: ModelSpec,
    suite: Union[str, Path, Suite] = "quick",
    sizes: Optional[Sequence[int]] = None,
    out_dir: Union[str, Path, None] = None,
    device: str = "auto",
    resume: bool = True,
    workers: int = 1,
    log: Callable[[str], None] = print,
) -> Path:
    """Benchmark one model on a suite. Returns the run directory.

    With ``workers > 1`` trials run in parallel CPU processes (one thread each); ``device="auto"``
    then means CPU.
    """
    suite = load_suite(suite)
    sizes = [int(s) for s in (sizes or suite.default_sizes)]
    out = Path(out_dir or Path("runs") / f"{spec.name}_{suite.name}")
    workers = max(1, int(workers))
    dev = torch.device("cpu") if (workers > 1 and device == "auto") else pick_device(device)

    old = read_meta(out)
    if old and resume and old.get("suite_fingerprint") not in (None, suite.fingerprint):
        raise ValueError(
            f"{out} holds results from a different suite configuration; use another --out "
            "directory or delete it."
        )

    # 1. validate the model contract for every size
    checks = []
    for size in sizes:
        info = check_model(spec, size, T=min(64, suite.T))
        checks.append(info)
        if not info["responds_to_input"]:
            log(f"warning: {spec.name} (width={size}) output does not react to its input")
    n_params = {c["width"]: c["n_params"] for c in checks}

    trials = plan_trials(suite, sizes)
    done = completed_keys(out) if resume else set()
    todo = [t for t in trials if trial_key(t) not in done]

    meta = {
        "model": spec.name,
        "model_source": spec.source,
        "model_kwargs": spec.kwargs,
        "sizes": sizes,
        "n_params": n_params,
        "causality_check": "passed",
        "suite": suite.to_dict(),
        "suite_name": suite.name,
        "suite_fingerprint": suite.fingerprint,
        "device": str(dev),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "cmf": __version__,
        "started": old.get("started") or _dt.datetime.now().isoformat(timespec="seconds"),
        "n_trials": len(trials),
    }
    write_meta(out, meta)

    log(f"[cmf] {spec.name} on suite '{suite.name}': {len(trials)} trials "
        f"({len(trials) - len(todo)} already done), sizes={sizes}, device={dev}, workers={workers}")
    log(f"[cmf] parameters per size: {n_params}")
    log(f"[cmf] writing to {out}/results.csv")

    t0 = time.perf_counter()

    def _done(i: int, row: dict) -> None:
        append_row(out, row)
        elapsed = time.perf_counter() - t0
        eta = elapsed / i * (len(todo) - i)
        log(f"[{i:>4}/{len(todo)}] {row['memory_function']:<12} a={row['alpha']:<10.4g} "
            f"m={row['size']:<4} seed={row['seed']:<3} rLoss={row['train_rloss_min']:.3e} "
            f"({row['epochs']} ep, {row['wall_time_s']:.1f}s)  eta {_fmt_time(eta)}")

    if workers > 1 and len(todo) > 1:
        try:
            pickle.dumps(spec)
        except Exception as exc:
            log(f"[cmf] model cannot be sent to worker processes ({exc}); running with 1 worker. "
                "Define the model class at module level to enable --workers.")
            workers = 1
    finished = 0
    if workers > 1 and len(todo) > 1:
        ctx = multiprocessing.get_context("spawn")
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                     initializer=_init_worker) as pool:
                futures = [pool.submit(run_trial, spec, t, suite, str(dev)) for t in todo]
                for fut in as_completed(futures):
                    finished += 1
                    _done(finished, fut.result())
        except BrokenProcessPool as exc:
            log(f"[cmf] worker processes failed ({exc}); continuing in a single process. "
                "(Parallel mode needs a script file or importable module, not stdin.)")
            done_now = completed_keys(out)
            todo_left = [t for t in todo if trial_key(t) not in done_now]
            for t in todo_left:
                finished += 1
                _done(finished, run_trial(spec, t, suite, str(dev)))
    else:
        for t in todo:
            finished += 1
            _done(finished, run_trial(spec, t, suite, str(dev)))

    meta["finished"] = _dt.datetime.now().isoformat(timespec="seconds")
    meta["wall_time_s"] = round(time.perf_counter() - t0 + float(old.get("wall_time_s", 0)), 1)
    write_meta(out, meta)
    log(f"[cmf] done in {_fmt_time(time.perf_counter() - t0)} -> {out}")
    return out
