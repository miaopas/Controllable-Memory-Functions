"""cmf: benchmark what kinds of memory a sequence model can learn.

Quick start (Python)::

    import torch.nn as nn
    import cmf

    class MyGRU(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.rnn = nn.GRU(width, width, batch_first=True)

        def forward(self, x):
            return self.rnn(x)[0]

    run = cmf.run(MyGRU, name="MyGRU", suite="quick", sizes=[8, 16, 32])
    lstm = cmf.run("lstm", suite="quick", sizes=[8, 16, 32])
    cmf.report([run, lstm], out="reports/mygru.html")

See ``python -m cmf --help`` for the command line.
"""

from __future__ import annotations

import sys
from pathlib import Path

__version__ = "0.1.0"

# The tool reuses the research code in ``src/`` (memory functions, data generator, models).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from typing import Callable, Optional, Sequence, Union  # noqa: E402

from torch import nn  # noqa: E402

from .spec import ModelSpec, check_model, from_callable, resolve_model  # noqa: E402
from .suite import Suite, available_suites, load_suite  # noqa: E402


def run(
    model: Union[str, Callable[[int], nn.Module], ModelSpec],
    *,
    name: Optional[str] = None,
    suite: Union[str, Suite] = "quick",
    sizes: Optional[Sequence[int]] = None,
    out_dir: Union[str, Path, None] = None,
    device: str = "auto",
    workers: int = 1,
    size_arg: Optional[str] = None,
    **model_kwargs,
) -> Path:
    """Benchmark a model and return its run directory.

    ``model`` can be a builtin name (``"lstm"``, ``"gru"``, ``"tcn"``, ``"transformer"``), an import
    path (``"pkg.module:MyModel"``), a class/function called as ``model(width, **model_kwargs)``
    (or ``model(**{size_arg: width}, **model_kwargs)``), or a :class:`ModelSpec`.

    ``workers > 1`` trains that many trials in parallel CPU processes (one thread each), which is
    usually the fastest option for the small models in the ``smoke``/``quick`` suites.
    """
    from .runner import run_benchmark

    if isinstance(model, ModelSpec):
        spec = model
    elif isinstance(model, str):
        spec = resolve_model(model, size_arg=size_arg, kwargs=model_kwargs, name=name)
    else:
        spec = from_callable(name or getattr(model, "__name__", "model"), model,
                             size_arg=size_arg, kwargs=model_kwargs)
    return run_benchmark(spec, suite=suite, sizes=sizes, out_dir=out_dir, device=device,
                         workers=workers)


def report(run_dirs: Sequence[Union[str, Path]], out: Union[str, Path] = "report.html",
           primary: Optional[str] = None, metric: Optional[str] = None) -> Path:
    """Build an HTML comparison report from one or more run directories."""
    from .report import build_report

    return build_report(run_dirs, out=out, primary=primary, metric=metric)


__all__ = [
    "run", "report", "ModelSpec", "Suite", "check_model", "resolve_model", "from_callable",
    "load_suite", "available_suites", "__version__",
]
