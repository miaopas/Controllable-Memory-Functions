"""Run trials as Ray tasks: one machine with several GPUs, or a Ray cluster.

Install with ``pip install -e ".[ray]"``. On a multi-node cluster the repository must be importable
on every node (shared filesystem + ``pip install -e .``, or a Ray ``runtime_env``).
"""

from __future__ import annotations

from typing import Callable, List, Optional


def _trial(spec, t, suite, device, threads):
    import torch

    from .runner import run_trial

    torch.set_num_threads(threads)
    return run_trial(spec, t, suite, device)


def run_trials_ray(spec, todo: List[dict], suite, on_done: Callable[[dict], None],
                   address: Optional[str] = None, cpus_per_trial: float = 1,
                   gpus_per_trial: float = 0, runtime_env: Optional[dict] = None) -> None:
    """Submit every trial as a Ray task and call ``on_done(row)`` as results arrive.

    ``gpus_per_trial`` may be fractional (e.g. 0.25 packs four small trials onto one GPU).
    """
    try:
        import ray
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError('The Ray backend needs Ray: pip install -e ".[ray]"') from exc

    if not ray.is_initialized():
        ray.init(address=address, runtime_env=runtime_env, log_to_driver=False,
                 include_dashboard=False)
    device = "cuda" if gpus_per_trial > 0 else "cpu"
    threads = max(1, int(cpus_per_trial))
    task = ray.remote(num_cpus=cpus_per_trial, num_gpus=gpus_per_trial)(_trial)
    pending = [task.remote(spec, t, suite, device, threads) for t in todo]
    while pending:
        ready, pending = ray.wait(pending, num_returns=1)
        on_done(ray.get(ready[0]))
