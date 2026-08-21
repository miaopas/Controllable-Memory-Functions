"""Model specification: how a user model is built, wrapped and validated.

The model contract
------------------
A user model is any ``torch.nn.Module`` built by ``factory(m)`` for a width ``m`` such that

* ``forward(x)`` maps ``(batch, T, m) -> (batch, T, m)`` (or returns a tuple whose first element
  has that shape, as ``torch.nn.LSTM`` does), and
* it is **causal**: the output at time ``t`` depends only on inputs at times ``<= t``.

The benchmark wraps it as ``Linear(1 -> m) -> model -> Linear(m -> 1)``, the same layout as the
``SequentialModel`` used by the paper configs.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Union

import torch
from torch import nn


class ModelCheckError(ValueError):
    """Raised when a model violates the model contract."""


class WrappedModel(nn.Module):
    """``Linear(1 -> m) -> core -> Linear(m -> 1)``."""

    def __init__(self, core: nn.Module, width: int, input_dim: int = 1, output_dim: int = 1):
        super().__init__()
        self.input = nn.Linear(input_dim, width)
        self.core = core
        self.output = nn.Linear(width, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.core(self.input(x))
        if isinstance(h, (tuple, list)):
            h = h[0]
        return self.output(h)


@dataclass
class ModelSpec:
    """Everything needed to build a model of a given width.

    The spec is picklable (no lambdas) so trials can run in worker processes.

    Args:
        name: Display name used in results and reports.
        target: What builds the *core* model (without the input/output projections): a builtin
            key ``"builtin:lstm"``, an import path ``"pkg.module:Class"``, or a callable.
        size_arg: Keyword that receives the width. ``None`` passes it as the first positional arg.
        kwargs: Extra constructor keyword arguments.
        source: Free-text description of where the model came from (shown in the report).
    """

    name: str
    target: Union[str, Callable[..., nn.Module]]
    size_arg: Optional[str] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)
    source: str = ""

    def factory(self, width: int) -> nn.Module:
        target = self.target
        if isinstance(target, str):
            if target.startswith("builtin:"):
                from .builtins import BUILTINS

                return BUILTINS[target.split(":", 1)[1]].factory(width, **self.kwargs)
            target = _import_object(target)
        if self.size_arg is None:
            return target(width, **self.kwargs)
        return target(**{self.size_arg: width}, **self.kwargs)

    def build(self, width: int) -> WrappedModel:
        core = self.factory(width)
        if not isinstance(core, nn.Module):
            raise ModelCheckError(
                f"{self.name}: factory({width}) returned {type(core).__name__}, expected nn.Module"
            )
        return WrappedModel(core, width)


def _import_object(path: str) -> Any:
    """Import ``"package.module:attr"`` (or ``"package.module.attr"``)."""
    if ":" in path:
        module_name, attr = path.split(":", 1)
    else:
        module_name, _, attr = path.rpartition(".")
    if not module_name:
        raise ValueError(f"Cannot import {path!r}: use 'package.module:ClassName'")
    module = importlib.import_module(module_name)
    obj = module
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def resolve_model(
    model: str,
    size_arg: Optional[str] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
) -> ModelSpec:
    """Turn a CLI-style model reference into a :class:`ModelSpec`.

    ``model`` is either a builtin baseline name (see ``cmf.builtins.BUILTINS``) or an import path
    ``package.module:ClassOrFunction``. The width is passed as keyword ``size_arg``; if
    ``size_arg`` is omitted it is passed as the first positional argument.
    """
    from .builtins import BUILTINS

    kwargs = dict(kwargs or {})
    key = model.lower()
    if key in BUILTINS:
        return ModelSpec(name=name or BUILTINS[key].display_name, target=f"builtin:{key}",
                         kwargs=kwargs, source=f"builtin:{key}")

    obj = _import_object(model)
    if not callable(obj):
        raise ValueError(f"{model!r} is not callable")
    default_name = getattr(obj, "__name__", model.split(":")[-1])
    return ModelSpec(name=name or default_name, target=model, size_arg=size_arg, kwargs=kwargs,
                     source=model)


def from_callable(name: str, model_fn: Callable[..., nn.Module], size_arg: Optional[str] = None,
                  kwargs: Optional[Dict[str, Any]] = None) -> ModelSpec:
    """Spec from a Python class/function called as ``model_fn(width, **kwargs)``."""
    if not callable(model_fn):
        raise TypeError("model_fn must be callable: model_fn(width) -> nn.Module")
    module = getattr(model_fn, "__module__", "")
    qual = getattr(model_fn, "__qualname__", repr(model_fn))
    return ModelSpec(name=name, target=model_fn, size_arg=size_arg, kwargs=dict(kwargs or {}),
                     source=f"{module}.{qual}" if module else qual)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def check_model(
    spec: ModelSpec,
    width: int,
    T: int = 64,
    batch: int = 2,
    t0: Optional[int] = None,
    atol: float = 1e-5,
) -> Dict[str, Any]:
    """Validate the model contract for one width. Returns a small report dict.

    * **shape**: ``(batch, T, 1) -> (batch, T, 1)`` after wrapping.
    * **causality**: perturbing the input at step ``t0`` must not change outputs at steps < ``t0``.

    Raises :class:`ModelCheckError` on failure.
    """
    model = spec.build(width).eval()
    x = torch.randn(batch, T, 1)
    try:
        y = model(x)
    except Exception as exc:  # pragma: no cover - message is what matters
        raise ModelCheckError(f"{spec.name} (width={width}) failed on input {tuple(x.shape)}: {exc}")

    if tuple(y.shape) != (batch, T, 1):
        raise ModelCheckError(
            f"{spec.name} (width={width}): expected output (batch, T, 1)={(batch, T, 1)} after "
            f"wrapping, got {tuple(y.shape)}. The core model must map (batch, T, m) -> (batch, T, m)."
        )

    t0 = T // 2 if t0 is None else t0
    x2 = x.clone()
    x2[:, t0, :] += 1.0
    y2 = model(x2)
    leak = (y2[:, :t0] - y[:, :t0]).abs().max().item()
    if leak > atol:
        raise ModelCheckError(
            f"{spec.name} (width={width}) is not causal: changing the input at t={t0} changed "
            f"earlier outputs by up to {leak:.3g}. Models must only use past and present inputs."
        )
    effect = (y2[:, t0:] - y[:, t0:]).abs().max().item()
    return {
        "width": width,
        "n_params": count_parameters(model),
        "causal": True,
        "max_leak": leak,
        "responds_to_input": effect > atol,
    }
