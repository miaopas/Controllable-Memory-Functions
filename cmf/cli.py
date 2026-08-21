"""Command line: ``python -m cmf {run,report,check,list}``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def _cmd_list(args) -> int:
    from .builtins import BUILTINS
    from .suite import available_suites, load_suite
    from src.data.lf.rho import BaseRho

    print("Suites:")
    for name in available_suites():
        s = load_suite(name)
        fns = ", ".join(f.name for f in s.functions)
        print(f"  {name:<10} {s.n_trials(len(s.default_sizes)):>5} trials (default sizes "
              f"{s.default_sizes}); T={s.T}; {fns}")
    print("\nBuiltin models (use as --model NAME):")
    for key, b in BUILTINS.items():
        print(f"  {key:<12} {b.description}")
    print("\nMemory functions available for custom suites:")
    print("  " + ", ".join(sorted(BaseRho.registry)))
    return 0


def _spec_from_args(args):
    from .spec import resolve_model

    kwargs = json.loads(args.model_kwargs) if args.model_kwargs else {}
    return resolve_model(args.model, size_arg=args.size_arg, kwargs=kwargs, name=args.name)


def _cmd_check(args) -> int:
    from .spec import ModelCheckError, check_model

    spec = _spec_from_args(args)
    ok = True
    for size in args.sizes or [8, 16, 32]:
        try:
            info = check_model(spec, size)
            print(f"ok  {spec.name} width={size}: {info['n_params']:,} params, causal")
        except ModelCheckError as e:
            ok = False
            print(f"FAIL {e}")
    return 0 if ok else 1


def _cmd_run(args) -> int:
    from .runner import run_benchmark

    spec = _spec_from_args(args)
    out = run_benchmark(spec, suite=args.suite, sizes=args.sizes, out_dir=args.out,
                        device=args.device, resume=not args.restart, workers=args.workers)
    if args.report:
        from .report import build_report

        runs = [out] + [Path(p) for p in args.compare]
        path = build_report(runs, out=args.report)
        print(f"[cmf] report: {path}")
    return 0


def _cmd_report(args) -> int:
    from .report import build_report

    path = build_report(args.runs, out=args.out, primary=args.primary, metric=args.metric)
    print(f"[cmf] report: {path}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m cmf",
        description="Benchmark what kinds of memory a sequence model can learn.",
    )
    p.add_argument("--version", action="version", version=f"cmf {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def model_args(sp):
        sp.add_argument("--model", required=True,
                        help="builtin name (lstm, gru, tcn, transformer) or import path 'pkg.module:Class'")
        sp.add_argument("--size-arg", default=None,
                        help="keyword that receives the width m (default: first positional argument)")
        sp.add_argument("--model-kwargs", default=None, help="extra constructor kwargs as JSON")
        sp.add_argument("--name", default=None, help="display name (default: class name)")
        sp.add_argument("--sizes", type=int, nargs="+", default=None, help="model widths m to test")

    sp = sub.add_parser("list", help="list suites, builtin models and memory functions")
    sp.set_defaults(func=_cmd_list)

    sp = sub.add_parser("check", help="check that a model satisfies the contract (shape, causality)")
    model_args(sp)
    sp.set_defaults(func=_cmd_check)

    sp = sub.add_parser("run", help="benchmark a model on a suite")
    model_args(sp)
    sp.add_argument("--suite", default="quick", help="smoke | quick | standard | path/to/suite.yaml")
    sp.add_argument("--out", default=None, help="run directory (default: runs/<name>_<suite>)")
    sp.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    sp.add_argument("--workers", type=int, default=1,
                    help="parallel CPU worker processes (try 6-8 on a laptop; 1 = single process)")
    sp.add_argument("--restart", action="store_true", help="ignore finished trials in --out")
    sp.add_argument("--report", default=None, help="also write an HTML report to this path")
    sp.add_argument("--compare", nargs="*", default=[], help="baseline run dirs for --report")
    sp.set_defaults(func=_cmd_run)

    sp = sub.add_parser("report", help="build an HTML comparison report from run directories")
    sp.add_argument("runs", nargs="+", help="run directories; the first is 'your model'")
    sp.add_argument("--out", default="report.html")
    sp.add_argument("--primary", default=None, help="model name to treat as 'your model'")
    sp.add_argument("--metric", default=None,
                    help="train_rloss_min (default) | val_rloss_min | train_loss_min | val_loss_min | test_rloss")
    sp.set_defaults(func=_cmd_report)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
