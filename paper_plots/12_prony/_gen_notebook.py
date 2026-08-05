"""
Generate paper_plots/12_prony/results.ipynb

Run: poetry run python paper_plots/12_prony/_gen_notebook.py
Then: poetry run jupyter nbconvert --to notebook --execute paper_plots/12_prony/results.ipynb --output results.ipynb
"""
import json
from pathlib import Path


def code_cell(lines):
    src = "\n".join(lines) + "\n"
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src}


def md_cell(lines):
    src = "\n".join(lines) + "\n"
    return {"cell_type": "markdown", "metadata": {}, "source": src}


cells = []

# --- title ---
cells.append(md_cell([
    "# Prony Series Spread Analysis",
    "",
    "**Condition 1** (Fig A): fix K=8, sweep γ — shows spread drives difficulty.",
    "**Condition 2** (Fig B): fix γ=0.5, sweep K — shows count alone does not matter.",
    "",
    "$$\\\\rho_{K,\\\\gamma}(s) = \\\\frac{1}{K}\\\\sum_{k=1}^{K} \\\\lambda_k^s, "
    "\\\\quad \\\\lambda_k = \\\\exp\\\\!\\\\left(-\\\\frac{10^{\\\\gamma(k-1)/(K-1)}}{T}\\\\right)$$",
]))

# --- imports & config ---
cells.append(code_cell([
    "import matplotlib",
    "matplotlib.use('Agg')",
    "import matplotlib.pyplot as plt",
    "import matplotlib.patches as mpatches",
    "import numpy as np",
    "import pandas as pd",
    "import autoroot",
    "from pathlib import Path",
    "from paper_plots.plot_config import set_plot_style",
    "from paper_plots.model_registry import MODEL_KEY_REGISTRY",
    "",
    "set_plot_style(font_family='Times New Roman', font_size=28, figsize=(10, 7), palette='tab20')",
    "",
    "RESULTS_DIR = Path('results')",
    "OUT_DIR     = Path('paper_plots/12_prony')",
    "OUT_DIR.mkdir(parents=True, exist_ok=True)",
    "",
    "LOSS_COL = 'train/rLoss_epoch.min'",
    "LW = 4",
    "",
    "ARCH_CFG = {",
    "    'LSTM': {'m_key': MODEL_KEY_REGISTRY['LSTM'],  'm_vals': [8, 16, 32, 64, 128]},",
    "    'S4':   {'m_key': MODEL_KEY_REGISTRY['S4'],    'm_vals': [4, 8, 16, 32, 64]},",
    "}",
    "",
    "def load(pkl_name):",
    "    path = RESULTS_DIR / pkl_name",
    "    if not path.exists():",
    "        print(f'  [missing] {path}')",
    "        return None",
    "    return pd.read_pickle(path)",
]))

# --- Figure A: loss vs gamma ---
cells.append(md_cell(["## Figure A — Loss vs γ (K=8 fixed)"]))

cells.append(code_cell([
    "fig, axes = plt.subplots(1, 2, figsize=(20, 8))",
    "",
    "for ax, (arch, cfg) in zip(axes, ARCH_CFG.items()):",
    "    df = load(f'{arch}_prony_gamma_experiment_data_clean.pkl')",
    "    if df is None:",
    "        ax.set_title(f'{arch} (no data)')",
    "        continue",
    "    df['m'] = df[cfg['m_key']]",
    "    df['gamma'] = df['data.rho_param.gamma']",
    "",
    "    m_vals = sorted(df['m'].unique())",
    "    colors = plt.cm.Accent(np.linspace(0, 1, len(m_vals)))",
    "    palette = dict(zip(m_vals, colors))",
    "",
    "    for m in m_vals:",
    "        curve = df[df['m'] == m].groupby('gamma')[LOSS_COL].min()",
    "        ax.plot(curve.index, curve.values, lw=LW, color=palette[m], label=str(int(m)))",
    "",
    "    ax.set_yscale('log')",
    "    ax.set_title(arch)",
    "    ax.set_xlabel(r'$\\gamma$  (0 = single exp, 1 = max spread)')",
    "    ax.set_ylabel('rLoss')",
    "    ax.grid(True, alpha=0.3)",
    "    ax.legend(title=r'$m$', fontsize=18)",
    "",
    "fig.suptitle('Prony Spread: rLoss vs γ (K=8 fixed)', fontsize=32)",
    "fig.tight_layout()",
    "out = OUT_DIR / 'prony_spread_vs_gamma.pdf'",
    "fig.savefig(out, dpi=150, bbox_inches='tight')",
    "print(f'Saved {out}')",
    "plt.show()",
]))

# --- Figure B: loss vs K ---
cells.append(md_cell(["## Figure B — Loss vs K (γ=0.5 fixed)"]))

cells.append(code_cell([
    "fig, axes = plt.subplots(1, 2, figsize=(20, 8))",
    "",
    "for ax, (arch, cfg) in zip(axes, ARCH_CFG.items()):",
    "    df = load(f'{arch}_prony_K_experiment_data_clean.pkl')",
    "    if df is None:",
    "        ax.set_title(f'{arch} (no data)')",
    "        continue",
    "    df['m'] = df[cfg['m_key']]",
    "    df['K'] = df['data.rho_param.K'].astype(int)",
    "",
    "    m_vals = sorted(df['m'].unique())",
    "    colors = plt.cm.Accent(np.linspace(0, 1, len(m_vals)))",
    "    palette = dict(zip(m_vals, colors))",
    "",
    "    for m in m_vals:",
    "        curve = df[df['m'] == m].groupby('K')[LOSS_COL].min()",
    "        ax.plot(curve.index, curve.values, lw=LW, color=palette[m],",
    "                label=str(int(m)), marker='o', markersize=8)",
    "",
    "    ax.set_xscale('log', base=2)",
    "    ax.set_yscale('log')",
    "    ax.set_title(arch)",
    "    ax.set_xlabel(r'$K$ (number of modes)')",
    "    ax.set_ylabel('rLoss')",
    "    ax.grid(True, alpha=0.3)",
    "    ax.legend(title=r'$m$', fontsize=18)",
    "",
    "fig.suptitle(r'Prony Count: rLoss vs K ($\\gamma=0.5$ fixed)', fontsize=32)",
    "fig.tight_layout()",
    "out = OUT_DIR / 'prony_spread_vs_K.pdf'",
    "fig.savefig(out, dpi=150, bbox_inches='tight')",
    "print(f'Saved {out}')",
    "plt.show()",
]))

# --- sanity check ---
cells.append(md_cell([
    "## Sanity checks",
    "",
    "- γ=0, K=8: all modes identical → loss should match single exponential at α≈1/T (very easy)",
    "- γ=1, K=8: loss should approach polynomial difficulty level",
    "- K=1 at γ=0.5: effectively single mode → baseline",
]))

cells.append(code_cell([
    "for arch, cfg in ARCH_CFG.items():",
    "    df = load(f'{arch}_prony_gamma_experiment_data_clean.pkl')",
    "    if df is None: continue",
    "    df['m'] = df[cfg['m_key']]",
    "    df['gamma'] = df['data.rho_param.gamma']",
    "    best_m = max(cfg['m_vals'])",
    "    sub = df[df['m'] == best_m]",
    "    curve = sub.groupby('gamma')[LOSS_COL].min()",
    "    print(f'{arch} (m={best_m}): gamma=0 loss={curve.get(0.0, float(\"nan\")):.4e},  gamma=1 loss={curve.get(1.0, float(\"nan\")):.4e}')",
]))

# --- write notebook ---
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

out_path = Path(__file__).parent / "results.ipynb"
out_path.write_text(json.dumps(notebook, indent=1))
print(f"Notebook written to {out_path}")
print("Execute with:")
print("  poetry run jupyter nbconvert --to notebook --execute paper_plots/12_prony/results.ipynb --output results.ipynb")
