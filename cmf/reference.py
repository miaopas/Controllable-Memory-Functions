"""Bundled reference results for the builtin models, so a new model can be compared without
re-training the baselines.

Layout: ``cmf/reference/<suite>/<builtin key>/{run.json,results.csv}``. They are produced with
``python -m cmf run --model <key> --suite <suite>`` and the suite's default sizes; see the README for
the hardware and time they took.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

REFERENCE_ROOT = Path(__file__).resolve().parent / "reference"


def available_references() -> Dict[str, List[str]]:
    """``{suite name: [builtin keys with bundled results]}``."""
    out: Dict[str, List[str]] = {}
    if REFERENCE_ROOT.is_dir():
        for suite_dir in sorted(p for p in REFERENCE_ROOT.iterdir() if p.is_dir()):
            keys = sorted(p.name for p in suite_dir.iterdir() if (p / "results.csv").exists())
            if keys:
                out[suite_dir.name] = keys
    return out


def reference_runs(suite: str, models: Union[str, Iterable[str], None] = "all") -> List[Path]:
    """Run directories of bundled baselines for ``suite`` (``models``: "all" or builtin keys)."""
    have = available_references().get(suite, [])
    if models in (None, "all"):
        wanted = have
    else:
        wanted = [m.strip().lower() for m in ([models] if isinstance(models, str) else models)]
        wanted = [k for m in wanted for k in (m.split(",") if "," in m else [m]) if k]
        missing = [m for m in wanted if m not in have]
        if missing:
            raise ValueError(f"no bundled '{suite}' results for {missing}; available: {have or 'none'}")
    return [REFERENCE_ROOT / suite / k for k in wanted]


def resolve_baselines(suite: Optional[str], baselines) -> List[Path]:
    if not baselines:
        return []
    if suite is None:
        raise ValueError("cannot pick bundled baselines without a suite name")
    return reference_runs(suite, baselines)
