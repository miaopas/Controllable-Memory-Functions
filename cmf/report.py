"""Self-contained HTML comparison report (figures embedded as PNG, plus data tables)."""

from __future__ import annotations

import base64
import datetime as _dt
import html
import io
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

from . import analysis as A  # noqa: E402
from .results import load_runs  # noqa: E402
from .suite import load_suite  # noqa: E402

# Reference categorical palette, fixed order: the primary model is always slot 1.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# Sequential blue ramp (100 -> 700) for magnitude (heatmaps) and ordered difficulty levels.
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#b9b8b3", "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2,
    "ytick.color": TEXT_2, "text.color": TEXT, "axes.titlecolor": TEXT, "axes.titlesize": 11,
    "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.family": "sans-serif", "lines.linewidth": 2,
    "legend.frameon": False,
})


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _nice_log(ax, axis: str = "y") -> None:
    """Log axis with plain-number tick labels (0.5, 1, 2 ... instead of 5x10^-1)."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    fmt = FuncFormatter(lambda v, _: f"{v:g}" if 1e-3 <= v < 1e4 else f"{v:.0e}".replace("e-0", "e-"))
    a = ax.yaxis if axis == "y" else ax.xaxis
    lo, hi = (ax.get_ylim() if axis == "y" else ax.get_xlim())
    span = np.log10(hi / lo) if lo > 0 and hi > 0 else 2.0
    subs = (1.0,) if span > 4 else (1.0, 2.0, 5.0) if span > 1.2 else tuple(np.arange(1.0, 10.0))
    a.set_major_locator(LogLocator(base=10, subs=subs))
    a.set_major_formatter(fmt)
    a.set_minor_formatter(NullFormatter())


def _colors(models: Sequence[str]) -> Dict[str, str]:
    return {m: SERIES[i % len(SERIES)] for i, m in enumerate(models)}


def _legend_top(fig, axes, models, colors):
    handles = [plt.Line2D([], [], color=colors[m], marker="o", markersize=5, lw=2) for m in models]
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.legend(handles, models, loc="upper center", ncol=min(len(models), 6),
               bbox_to_anchor=(0.5, 1.0))


# --------------------------------------------------------------------------- figures
def fig_memory_functions(suite) -> str:
    from src.data.lf.rho import BaseRho

    fns = suite.functions
    fig, axes = plt.subplots(1, len(fns), figsize=(3.1 * len(fns), 2.6), squeeze=False)
    for ax, fn in zip(axes[0], fns):
        T = suite.T_for(fn)
        t = torch_arange(T)
        k = len(fn.alpha)
        steps = np.linspace(3, len(BLUE_RAMP) - 1, k).round().astype(int)
        for i, a in enumerate(sorted(fn.alpha)):
            rho = BaseRho.create(fn.name, dim=1, T=T, alpha=a, **fn.params)
            vals = np.asarray(rho(t))[0]
            ax.plot(np.arange(T), vals, color=BLUE_RAMP[steps[i]], lw=1.6,
                    label=f"α={a:.3g}")
        if fn.name in ("exponential", "polynomial"):
            ax.set_yscale("symlog", linthresh=1e-6)
            ax.set_ylim(bottom=0)
        ax.set_title(A.label(fn.name))
        ax.set_xlabel("lag s")
    axes[0][0].set_ylabel("ρ(s)")
    fig.text(0.5, -0.06, "light → dark = easy → hard (increasing α)", ha="center",
             color=TEXT_2, fontsize=9)
    fig.tight_layout()
    return _png(fig)


def torch_arange(T: int):
    import torch

    return torch.arange(T, dtype=torch.int64).repeat(1, 1)


def fig_profile(profile: pd.DataFrame, models: List[str], functions: List[str], metric: str) -> str:
    vals = profile["value"][profile["value"] > 0]
    vmin, vmax = (vals.min(), vals.max()) if len(vals) else (1e-6, 1)
    norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10))
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    n_levels = int(profile["level"].max())
    fig, axes = plt.subplots(1, len(models), figsize=(max(0.8 * n_levels, 2.4) * len(models) + 2.6, 0.55 * len(functions) + 1.6),
                             squeeze=False, sharey=True)
    for ax, m in zip(axes[0], models):
        grid = np.full((len(functions), n_levels), np.nan)
        for r in profile[profile["model"] == m].itertuples():
            if r.memory_function in functions:
                grid[functions.index(r.memory_function), r.level - 1] = r.value
        im = ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid[i, j]
                if np.isfinite(v):
                    dark = norm(v) > 0.55
                    ax.text(j, i, f"{v:.1g}", ha="center", va="center", fontsize=7.5,
                            color="white" if dark else TEXT)
        ax.set_xticks(range(n_levels), [str(i + 1) for i in range(n_levels)])
        ax.set_yticks(range(len(functions)), [A.label(f) for f in functions])
        ax.set_title(m, fontweight="bold" if m == models[0] else "normal")
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.9, pad=0.02)
    cb.set_label("loss (log scale)", fontsize=8.5)
    cb.outline.set_visible(False)
    _nice_log(cb.ax)
    fig.supxlabel("difficulty level (1 = easiest)", fontsize=9.5, color=TEXT_2, y=-0.02)
    return _png(fig)


def fig_sensitivity(sens: pd.DataFrame, models: List[str], functions: List[str], colors) -> str:
    fig, ax = plt.subplots(figsize=(1.3 * len(functions) * max(1, len(models) / 2) + 2, 3.2))
    width = 0.8 / len(models)
    for i, m in enumerate(models):
        xs, heights, bottoms = [], [], []
        for j, fn in enumerate(functions):
            s = sens[(sens["model"] == m) & (sens["memory_function"] == fn)]
            if s.empty:
                continue
            f = max(1 + float(s["score"].iloc[0]), 1e-3)
            xs.append(j - 0.4 + width * (i + 0.5))
            bottoms.append(min(1.0, f))
            heights.append(abs(f - 1.0) + 1e-9)
        ax.bar(xs, heights, bottom=bottoms, width=width * 0.86, color=colors[m], label=m,
               edgecolor=SURFACE, linewidth=1)
    ax.set_yscale("log")
    _nice_log(ax)
    ax.axhline(1, color=TEXT_2, lw=1)
    ax.set_xticks(range(len(functions)), [A.label(f) for f in functions])
    ax.set_ylabel("loss(hardest α) / loss(easiest α)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.legend(loc="upper center", ncol=min(len(models), 6), bbox_to_anchor=(0.5, 1.0))
    return _png(fig)


def _small_multiples(functions):
    n = len(functions)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 2.9), squeeze=False)
    return fig, axes[0]


def fig_loss_vs_alpha(best: pd.DataFrame, models, functions, colors, metric) -> str:
    fig, axes = _small_multiples(functions)
    for ax, fn in zip(axes, functions):
        for m in models:
            g = best[(best["model"] == m) & (best["memory_function"] == fn)]
            if g.empty:
                continue
            curve = g.groupby("alpha")["value"].min()
            ax.plot(curve.index, curve.values, color=colors[m], marker="o", markersize=4.5,
                    lw=2.4 if m == models[0] else 1.8, zorder=3 if m == models[0] else 2)
        ax.set_yscale("log")
        _nice_log(ax)
        ax.set_title(A.label(fn))
        ax.set_xlabel("α (memory difficulty)")
    axes[0].set_ylabel("best loss over sizes")
    _legend_top(fig, axes, models, colors)
    return _png(fig)


def fig_loss_vs_size(best: pd.DataFrame, models, functions, colors) -> str:
    fig, axes = _small_multiples(functions)
    for ax, fn in zip(axes, functions):
        for m in models:
            g = best[(best["model"] == m) & (best["memory_function"] == fn)]
            if g.empty:
                continue
            hard = g[g["alpha"] == g["alpha"].max()].sort_values("size")
            ax.plot(hard["size"], hard["value"], color=colors[m], marker="o", markersize=4.5,
                    lw=2.4 if m == models[0] else 1.8)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(best["size"].unique()), [str(v) for v in sorted(best["size"].unique())])
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_yscale("log")
        _nice_log(ax)
        ax.set_title(f"{A.label(fn)}, hardest α")
        ax.set_xlabel("model width m")
    axes[0].set_ylabel("best loss over seeds")
    _legend_top(fig, axes, models, colors)
    return _png(fig)


# --------------------------------------------------------------------------- html helpers
def _md(text: str) -> str:
    t = html.escape(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    return re.sub(r"\*(.+?)\*", r"<em>\1</em>", t)


def _num(v, fmt="{:.3g}") -> str:
    try:
        return "–" if v is None or not np.isfinite(v) else fmt.format(v)
    except TypeError:
        return html.escape(str(v))


def _fmt_secs(s: float) -> str:
    m, sec = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h} h {m:02d} min" if h else (f"{m} min {sec:02d} s" if m else f"{sec} s")


def _table(headers: List[str], rows: List[List[str]], cls: str = "") -> str:
    th = "".join(f"<th>{h}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tbl"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>'


def _img(b64: str, alt: str) -> str:
    return f'<img src="data:image/png;base64,{b64}" alt="{html.escape(alt)}">'


CSS = """
:root { color-scheme: light; --surface:#fcfcfb; --surface-2:#f3f2ef; --text:#0b0b0b;
  --text-2:#52514e; --border:#e4e3df; --accent:#2a78d6; }
* { box-sizing: border-box; }
body { margin:0; background:var(--surface); color:var(--text);
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 28px; margin: 0 0 4px; letter-spacing:-0.01em; }
h2 { font-size: 20px; margin: 40px 0 8px; padding-top: 8px; border-top: 1px solid var(--border); }
h3 { font-size: 16px; margin: 20px 0 6px; }
p, li { max-width: 78ch; }
.sub { color: var(--text-2); margin: 0 0 20px; }
.cards { display:grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap:12px; margin: 16px 0; }
.card { background: var(--surface-2); border-radius: 10px; padding: 12px 14px; }
.card .k { color: var(--text-2); font-size: 12.5px; text-transform: uppercase; letter-spacing: .04em; }
.card .v { font-size: 16px; font-weight: 600; margin-top: 2px; }
.swatch { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; vertical-align:baseline; }
.findings li { margin: 6px 0; }
img { max-width: 100%; height: auto; display:block; margin: 10px 0; }
.tbl { overflow-x: auto; margin: 8px 0 16px; }
table { border-collapse: collapse; font-size: 13.5px; font-variant-numeric: tabular-nums; }
th, td { padding: 6px 10px; border-bottom: 1px solid var(--border); text-align: right; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--text-2); font-weight: 600; }
details { margin: 8px 0; } summary { cursor: pointer; color: var(--accent); }
.note { color: var(--text-2); font-size: 13.5px; }
code { background: var(--surface-2); padding: 1px 5px; border-radius: 4px; font-size: 13px; }
pre { background: var(--surface-2); padding: 12px; border-radius: 8px; overflow-x:auto; font-size: 13px; }
"""


# --------------------------------------------------------------------------- report
def build_report(run_dirs: Sequence[Union[str, Path]], out: Union[str, Path] = "report.html",
                 primary: Optional[str] = None, metric: Optional[str] = None) -> Path:
    df, metas = load_runs(run_dirs)
    models = list(dict.fromkeys(m["model"] for m in metas))  # order of the given run dirs
    if primary:
        if primary not in models:
            raise ValueError(f"primary model {primary!r} not among {models}")
        models.remove(primary)
        models.insert(0, primary)
    primary = models[0]

    fingerprints = {m.get("suite_fingerprint") for m in metas}
    suite = load_suite(dict(metas[0]["suite"], functions={
        f["name"]: {"alpha": f["alpha"], **({"T": f["T"]} if f.get("T") else {}), **f.get("params", {})}
        for f in metas[0]["suite"]["functions"]}))
    metric = metric or suite.metric
    colors = _colors(models)
    functions = [f.name for f in suite.functions if f.name in set(df["memory_function"])]

    best = A.best_over_seeds(df, metric)
    profile = A.memory_profile(best)
    sens = A.sensitivity(best)
    scal = A.scaling(best)
    notes = A.findings(best, sens, scal, primary, models, df)
    counts = A.n_trials(df)
    meta_by_model = {m["model"]: m for m in metas}

    # ---- header + summary cards
    when = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = [f"<h1>Memory benchmark: {html.escape(primary)}</h1>",
             f'<p class="sub">Suite <strong>{html.escape(suite.name)}</strong> · '
             f"{len(models)} model{'s' if len(models) > 1 else ''} · generated {when}</p>"]
    if len(fingerprints) > 1:
        parts.append('<p class="note"><strong>Warning:</strong> these runs used different suite '
                     "settings; comparisons may not be like-for-like.</p>")

    cards = []
    for m in models:
        meta, c = meta_by_model[m], counts.get(m, {"total": 0, "ok": 0, "wall_time_s": 0})
        params = meta.get("n_params", {})
        prange = (f"{min(params.values()):,}–{max(params.values()):,}" if params else "?")
        cards.append(
            f'<div class="card"><div class="k"><span class="swatch" style="background:{colors[m]}"></span>'
            f'{"your model" if m == primary else "baseline"}</div><div class="v">{html.escape(m)}</div>'
            f'<div class="note">widths {", ".join(map(str, meta.get("sizes", [])))} · {prange} params<br>'
            f'{c["ok"]}/{meta.get("n_trials", c["total"])} trials · {_fmt_secs(c["wall_time_s"])} on {meta.get("device", "?")}<br>'
            f'causality check: {meta.get("causality_check", "?")}</div></div>')
    parts.append('<div class="cards">' + "".join(cards) + "</div>")

    # ---- findings
    parts.append("<h2>Key findings</h2>")
    parts.append('<ul class="findings">' + "".join(f"<li>{_md(n)}</li>" for n in notes) + "</ul>")
    if suite.name in ("smoke", "quick"):
        parts.append(f'<p class="note">The <code>{suite.name}</code> suite is small (few sizes, seeds and '
                     "epochs), so treat these findings as indicative. Use the <code>standard</code> suite "
                     "for publishable numbers.</p>")

    # ---- the task
    parts.append("<h2>What is being measured</h2>")
    parts.append(
        "<p>Each model reads a random input sequence <em>u = tanh(x)</em> (<em>x</em> Gaussian noise) and must "
        "output, at every step <em>t</em>, the target <em>y<sub>t</sub> = tanh(Σ<sub>s≤t</sub> ρ(s) · "
        "u<sub>t−s</sub>)</em>. "
        "The memory function ρ decides which past inputs matter. Each function has a difficulty "
        "parameter α; larger α means longer or harder memory. Below are the memory functions in this "
        "suite at the difficulty levels tested.</p>")
    parts.append(_img(fig_memory_functions(suite), "Memory functions at each tested difficulty"))

    # ---- profile
    parts.append("<h2>Memory profile</h2>")
    parts.append(f"<p>Best loss over model sizes and seeds for every memory function (rows) and "
                 f"difficulty level (columns). Lighter is better. Metric: {html.escape(A.METRICS.get(metric, metric))}.</p>")
    parts.append(_img(fig_profile(profile, models, functions, metric), "Memory profile heatmaps"))

    # ---- sensitivity
    parts.append("<h2>Difficulty sensitivity</h2>")
    parts.append("<p>How many times larger the loss is at the hardest α than at the easiest α "
                 "(geometric mean over model sizes, the same score as the paper). Lower bars mean the "
                 "model keeps up as the memory gets harder. A value of 1 means difficulty has no effect.</p>")
    parts.append(_img(fig_sensitivity(sens, models, functions, colors), "Difficulty sensitivity bars"))
    rows = []
    for fn in functions:
        row = [A.label(fn)]
        for m in models:
            s = sens[(sens["model"] == m) & (sens["memory_function"] == fn)]
            row.append("–" if s.empty else f"×{_num(1 + s['score'].iloc[0], '{:.3g}')}")
        rows.append(row)
    parts.append(_table(["memory function"] + [html.escape(m) for m in models], rows))

    # ---- curves
    parts.append("<h2>Loss vs difficulty</h2>")
    parts.append("<p>Best loss (over sizes and seeds) as α increases.</p>")
    parts.append(_img(fig_loss_vs_alpha(best, models, functions, colors, metric), "Loss versus alpha"))
    if best["size"].nunique() > 1:
        parts.append("<h2>Does a bigger model help?</h2>")
        parts.append("<p>Loss at the hardest α as the model width grows. A falling line means the "
                     "difficulty is a capacity problem; a flat line points to the architecture itself.</p>")
        parts.append(_img(fig_loss_vs_size(best, models, functions, colors), "Loss versus model size"))

    # ---- data tables
    parts.append("<h2>Data</h2>")
    rows = []
    for r in best.sort_values(["memory_function", "model", "alpha", "size"]).itertuples():
        rows.append([A.label(r.memory_function), html.escape(r.model), _num(r.alpha, "{:.4g}"),
                     str(r.size), f"{int(r.n_params):,}", _num(r.value), str(r.n_seeds)])
    parts.append("<details><summary>All results (best over seeds)</summary>"
                 + _table(["memory function", "model", "α", "width", "params", metric, "seeds"], rows)
                 + "</details>")

    # ---- method
    tr = suite.training
    d = suite.data
    parts.append("<h2>Setup</h2>")
    parts.append(_table(["setting", "value"], [
        ["suite", f"{html.escape(suite.name)} <span class='note'>({suite.fingerprint})</span>"],
        ["sequence length T", ", ".join(f"{A.label(f.name)} {suite.T_for(f)}" for f in suite.functions)],
        ["samples (train / val / test)", f"{d.n_train} / {d.n_val} / {d.n_test}"],
        ["seeds", ", ".join(map(str, suite.seeds))],
        ["optimiser", f"Adam, lr {tr.lr:g}, {tr.scheduler} schedule"],
        ["epochs", f"up to {tr.epochs} (min {tr.min_epochs}, early stop after {tr.patience} without improvement)"],
        ["batch size", str(tr.batch_size)],
        ["model wrapper", "Linear(1→m) → model → Linear(m→1)"],
    ]))
    for m in models:
        meta = meta_by_model[m]
        parts.append(f"<p class='note'><span class='swatch' style='background:{colors[m]}'></span>"
                     f"<strong>{html.escape(m)}</strong>: <code>{html.escape(str(meta.get('model_source', '')))}</code>"
                     + (f" kwargs <code>{html.escape(str(meta.get('model_kwargs')))}</code>" if meta.get("model_kwargs") else "")
                     + f" · run dir <code>{html.escape(meta['run_dir'])}</code></p>")
    parts.append("<p class='note'>Generated by <code>cmf</code> (Controllable Memory Functions).</p>")

    page = ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>Memory benchmark: {html.escape(primary)}</title><style>{CSS}</style></head>"
            f"<body><main>{''.join(parts)}</main></body></html>")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out
