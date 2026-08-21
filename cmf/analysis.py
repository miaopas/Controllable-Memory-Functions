"""Turn per-trial results into comparisons: memory profile, sensitivity score, scaling, findings."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

METRICS = {
    "train_rloss_min": "best training relative MSE (MSE / Var(y))",
    "val_rloss_min": "best validation relative MSE",
    "train_loss_min": "best training MSE",
    "val_loss_min": "best validation MSE",
    "test_rloss": "test relative MSE (final model)",
}

FUNCTION_LABELS = {
    "exponential": "Exponential",
    "polynomial": "Polynomial",
    "impulse": "Impulse",
    "airy": "Airy",
    "prony": "Prony",
    "mix_exp_impulse": "Exp + impulse mix",
}

FUNCTION_MEANING = {
    "exponential": "smoothly fading memory (short time-scale)",
    "polynomial": "heavy-tailed, slowly fading memory",
    "impulse": "exact recall of one input far in the past",
    "airy": "oscillating memory that reaches further back as α grows",
    "prony": "a mixture of many time-scales",
    "mix_exp_impulse": "fading memory mixed with a single delayed recall",
}


def label(fn: str) -> str:
    return FUNCTION_LABELS.get(fn, fn)


def best_over_seeds(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Min over seeds for every (model, function, alpha, size) - the paper's aggregation."""
    ok = df[df["status"] == "ok"]
    keys = ["model", "memory_function", "alpha", "alpha_norm", "size"]
    out = ok.groupby(keys, as_index=False).agg(
        value=(metric, "min"), n_params=("n_params", "first"), n_seeds=("seed", "nunique"))
    # difficulty level 1..k within each function (alpha grids differ between functions)
    out["level"] = out.groupby(["model", "memory_function"])["alpha"].rank(method="dense").astype(int)
    return out


def memory_profile(best: pd.DataFrame) -> pd.DataFrame:
    """Best value over sizes per (model, function, difficulty level)."""
    return best.groupby(["model", "memory_function", "level"], as_index=False).agg(
        value=("value", "min"), alpha=("alpha", "first"), alpha_norm=("alpha_norm", "first"))


def sensitivity(best: pd.DataFrame) -> pd.DataFrame:
    """Difficulty-sensitivity score per (model, function).

    For each size m: rel(m) = (L(alpha_max) - L(alpha_min)) / L(alpha_min). The score is the
    geometric mean of |rel(m)| over sizes, signed by the mean (as in
    ``paper_plots/*/calculate_alpha_sensitivity.py``). Larger = degrades more as memory gets harder.
    """
    rows = []
    for (model, fn), g in best.groupby(["model", "memory_function"]):
        lo, hi = g["alpha"].min(), g["alpha"].max()
        l_lo = g[g["alpha"] == lo].set_index("size")["value"]
        l_hi = g[g["alpha"] == hi].set_index("size")["value"]
        rel = ((l_hi - l_lo) / l_lo).dropna()
        rel = rel[np.isfinite(rel)]
        if rel.empty:
            continue
        nonzero = np.abs(rel.values)[np.abs(rel.values) > 0]
        gm = float(np.exp(np.log(nonzero).mean())) if len(nonzero) else 0.0
        rows.append({
            "model": model, "memory_function": fn,
            "score": gm * float(np.sign(rel.mean())),
            "easy_loss": float(l_lo.min()), "hard_loss": float(l_hi.min()),
            "n_sizes": int(len(rel)),
        })
    return pd.DataFrame(rows)


def scaling(best: pd.DataFrame) -> pd.DataFrame:
    """At the hardest alpha: how much does going from the smallest to largest size help?"""
    rows = []
    for (model, fn), g in best.groupby(["model", "memory_function"]):
        hard = g[g["alpha"] == g["alpha"].max()].sort_values("size")
        if len(hard) < 2:
            continue
        small, large = hard.iloc[0], hard.iloc[-1]
        rows.append({
            "model": model, "memory_function": fn,
            "size_small": int(small["size"]), "size_large": int(large["size"]),
            "loss_small": float(small["value"]), "loss_large": float(large["value"]),
            "gain": float(small["value"] / large["value"]) if large["value"] > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def _ratio_text(r: float) -> str:
    if r >= 1:
        return f"{r:.1f}×" if r < 10 else f"{r:.0f}×"
    return f"{1 / r:.1f}×" if 1 / r < 10 else f"{1 / r:.0f}×"


def findings(best: pd.DataFrame, sens: pd.DataFrame, scal: pd.DataFrame, primary: str,
             models: List[str], df: Optional[pd.DataFrame] = None) -> List[str]:
    """Rule-based plain-language observations about the primary model."""
    out: List[str] = []
    baselines = [m for m in models if m != primary]
    if sens.empty:
        return ["Not enough finished trials to compare difficulty levels yet."]
    s = sens[sens["model"] == primary].set_index("memory_function")
    if s.empty:
        return ["Not enough finished trials to compare difficulty levels yet."]

    # 1. which memory is easy / hard for the primary model
    order = s["score"].sort_values()
    easiest, hardest = order.index[0], order.index[-1]
    if len(order) > 1:
        out.append(
            f"**{primary}** copes best with *{label(easiest)}* memory ({FUNCTION_MEANING.get(easiest, '')}): "
            f"loss changes {_fmt_change(order.iloc[0])} from the easiest to the hardest setting. "
            f"It struggles most with *{label(hardest)}* memory ({FUNCTION_MEANING.get(hardest, '')}): "
            f"loss changes {_fmt_change(order.iloc[-1])}."
        )

    # 2. comparison with each baseline on hard settings
    for fn in s.index:
        hard_p = s.loc[fn, "hard_loss"]
        comps = []
        for b in baselines:
            sb = sens[(sens["model"] == b) & (sens["memory_function"] == fn)]
            if sb.empty:
                continue
            hard_b = float(sb["hard_loss"].iloc[0])
            if hard_p <= 0 or hard_b <= 0:
                continue
            r = hard_b / hard_p
            if r >= 1.5:
                comps.append(f"{_ratio_text(r)} lower than {b}")
            elif r <= 1 / 1.5:
                comps.append(f"{_ratio_text(r)} higher than {b}")
            else:
                comps.append(f"on par with {b}")
        if comps:
            out.append(f"*{label(fn)}*, hardest setting: {primary}'s best loss is "
                       + ", ".join(comps) + ".")

    # 3. does size help?
    sc = scal[scal["model"] == primary] if not scal.empty else scal
    if not sc.empty:
        helps = sc[sc["gain"] >= 2]
        flat = sc[sc["gain"] < 1.5]
        if not helps.empty:
            parts = [f"*{label(r.memory_function)}* ({_ratio_text(r.gain)})" for r in helps.itertuples()]
            r0 = helps.iloc[0]
            out.append(f"Making {primary} wider (m={r0.size_small} → {r0.size_large}) clearly helps on "
                       "the hardest setting of " + ", ".join(parts) + ".")
        if not flat.empty:
            out.append(f"Width barely helps {primary} on the hardest setting of "
                       + ", ".join(f"*{label(f)}*" for f in flat["memory_function"])
                       + "; the limitation looks architectural rather than a capacity issue.")

    # 4. failures
    if df is not None:
        bad = df[(df["model"] == primary) & (df["status"] != "ok")]
        if len(bad):
            out.append(f"{len(bad)} of {int((df['model'] == primary).sum())} trials of "
                       f"{primary} did not finish normally ({', '.join(sorted(bad['status'].unique()))}).")
    return out


def _fmt_change(score: float) -> str:
    """Express a relative increase (L_hard - L_easy) / L_easy as a factor."""
    factor = 1 + score
    if factor <= 0:
        return "very little"
    if factor < 1:
        return f"by ×{factor:.2f} (it actually gets easier)"
    if factor < 1.5:
        return f"by only ×{factor:.2f}"
    return f"by ×{factor:.1f}" if factor < 100 else f"by ×{factor:.0e}".replace("e+0", "e")


def summary_table(best: pd.DataFrame, sens: pd.DataFrame) -> pd.DataFrame:
    """Wide table: rows = function, columns = model, cells = score / easy / hard."""
    return sens.pivot_table(index="memory_function", columns="model",
                            values=["easy_loss", "hard_loss", "score"])


def n_trials(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    out = {}
    for m, g in df.groupby("model"):
        out[m] = {"total": int(len(g)), "ok": int((g["status"] == "ok").sum()),
                  "wall_time_s": float(g["wall_time_s"].sum())}
    return out
