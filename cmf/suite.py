"""Benchmark suites: which memory functions, difficulty grid, data size and training budget."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

SUITE_DIR = Path(__file__).parent / "suites"


@dataclass(frozen=True)
class DataConfig:
    n_train: int = 4096
    n_val: int = 512
    n_test: int = 512
    input_generator: str = "randn"
    inner_activation: str = "tanh"
    output_activation: str = "tanh"


@dataclass
class TrainConfig:
    epochs: int = 30
    min_epochs: int = 5
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    scheduler: str = "cosine"  # "cosine" (CosineAnnealingLR over `epochs`) or "none"
    patience: int = 5  # early stopping on the epoch-mean training loss, as in the paper configs
    grad_clip: Optional[float] = None


@dataclass
class FunctionConfig:
    name: str
    alpha: List[float]
    T: Optional[int] = None  # overrides the suite-level T
    params: Dict[str, Any] = field(default_factory=dict)  # extra fixed rho parameters


@dataclass
class Suite:
    name: str
    T: int
    functions: List[FunctionConfig]
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    seeds: List[int] = field(default_factory=lambda: [11])
    default_sizes: List[int] = field(default_factory=lambda: [8, 16, 32])
    metric: str = "train_rloss_min"
    description: str = ""

    def T_for(self, fn: FunctionConfig) -> int:
        return fn.T or self.T

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        """Short hash of everything that affects results (used to refuse mixing suites)."""
        d = self.to_dict()
        d.pop("description", None)
        d.pop("default_sizes", None)
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:10]

    def n_trials(self, n_sizes: int) -> int:
        return sum(len(f.alpha) for f in self.functions) * n_sizes * len(self.seeds)


def available_suites() -> List[str]:
    return sorted(p.stem for p in SUITE_DIR.glob("*.yaml"))


def load_suite(suite: Union[str, Path, Dict[str, Any], Suite]) -> Suite:
    """Load a suite by builtin name (``"quick"``), YAML path, or dict."""
    if isinstance(suite, Suite):
        return suite
    if isinstance(suite, dict):
        raw = dict(suite)
    else:
        path = Path(suite)
        if not path.suffix:
            path = SUITE_DIR / f"{suite}.yaml"
        if not path.exists():
            raise FileNotFoundError(
                f"Suite {suite!r} not found. Builtin suites: {', '.join(available_suites())}"
            )
        raw = yaml.safe_load(path.read_text())

    functions = []
    for name, cfg in raw.pop("functions").items():
        cfg = dict(cfg)
        alpha = [float(a) for a in cfg.pop("alpha")]
        T = cfg.pop("T", None)
        functions.append(FunctionConfig(name=name, alpha=alpha, T=T, params=cfg))

    return Suite(
        name=raw.pop("name", "custom"),
        T=int(raw.pop("T")),
        functions=functions,
        data=DataConfig(**raw.pop("data", {})),
        training=TrainConfig(**raw.pop("training", {})),
        seeds=[int(s) for s in raw.pop("seeds", [11])],
        default_sizes=[int(s) for s in raw.pop("default_sizes", [8, 16, 32])],
        metric=raw.pop("metric", "train_rloss_min"),
        description=" ".join(str(raw.pop("description", "")).split()),
        **raw,
    )
