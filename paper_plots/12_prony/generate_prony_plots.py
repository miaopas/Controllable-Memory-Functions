"""Prony-series difficulty plots.

The Prony target is rho(s) = (1/K) * sum_k lambda_k^s, with
lambda_k = exp(-10^(gamma*(k-1)/(K-1)) / T)  (see src/data/lf/rho.py:291).
Two knobs are swept independently:

  * gamma (spread): timescale spread, gamma=0 -> single exponential (easy),
    gamma=1 -> many timescales / polynomial-like (hard). Swept with K=8 fixed.
  * K (count): number of modes, swept [1,2,4,8,16,32] with gamma=0.5 fixed.
    Tests whether mode count alone drives difficulty.

Following the main-text combined plots (paper_plots/1_<MODEL>), we produce one
figure per model with two panels (spread sweep | count sweep), one curve per
model size m. Only LSTM and S4 have Prony runs.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import autoroot
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from pathlib import Path
from paper_plots.plot_config import set_plot_style
from paper_plots.model_registry import MODEL_KEY_REGISTRY

# --- Global styling (matches the 1_<MODEL> combined plots) ---
font_size = 40
figsize = (12, 8)
lw = 4
loss = 'train/rLoss_epoch.min'

set_plot_style(font_family='Times New Roman', font_size=font_size,
               figsize=figsize, palette='tab20')

# Per-model size-parameter key (same registry the other plots use).
MODEL_CONFIGS = {
    'LSTM': MODEL_KEY_REGISTRY['LSTM'],
    'S4': MODEL_KEY_REGISTRY['S4'],
}

results_dir = Path("results")
output_dir = Path("paper_plots/12_prony")


def load(model_name, sweep):
    """Load a Prony sweep pkl ('gamma' or 'K'), or None if missing."""
    path = results_dir / f"{model_name}_prony_{sweep}_experiment_data_clean.pkl"
    if not path.exists():
        print(f"  [skip] missing file: {path.name}")
        return None
    return pd.read_pickle(path)


def build_model_figure(model_name, m_key):
    print(f"[{model_name}] building prony figure...")
    df_g = load(model_name, 'gamma')
    df_k = load(model_name, 'K')
    if df_g is None and df_k is None:
        print(f"[{model_name}] no data found, skipping.\n")
        return

    # Per-(m, knob) minimum loss over seeds (no-op when there is a single seed).
    def reduce_min(df, knob_col, knob_name):
        df = df.copy()
        df['m'] = df[m_key]
        df[knob_name] = df[knob_col]
        return df.groupby(['m', knob_name], as_index=False)[loss].min()

    gmin = reduce_min(df_g, 'data.rho_param.gamma', 'gamma') if df_g is not None else None
    kmin = reduce_min(df_k, 'data.rho_param.K', 'K') if df_k is not None else None
    if kmin is not None:
        kmin['K'] = kmin['K'].astype(int)

    # One Accent color per model size, consistent across both panels + legend.
    m_values = sorted(set(
        (gmin['m'].unique().tolist() if gmin is not None else []) +
        (kmin['m'].unique().tolist() if kmin is not None else [])
    ))
    accent_colors = plt.cm.Accent(np.linspace(0, 1, len(m_values)))
    palette_dict = dict(zip(m_values, accent_colors))

    fig = plt.figure(figsize=(24, 12))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 0.15],
                  hspace=0.4, wspace=0.3)

    # Left panel: spread sweep (loss vs gamma, K=8 fixed).
    ax1 = fig.add_subplot(gs[0, 0])
    if gmin is not None:
        sns.lineplot(data=gmin, x='gamma', y=loss, hue='m', ax=ax1,
                     palette=palette_dict, lw=lw, legend=False)
        ax1.set_yscale('log')
    else:
        ax1.text(0.5, 0.5, 'no data', ha='center', va='center',
                 transform=ax1.transAxes, fontsize=font_size, color='gray')
    ax1.set_title(r'Spread ($K=8$)')
    ax1.set_xlabel(r'$\gamma$')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)

    # Right panel: count sweep (loss vs K, gamma=0.5 fixed).
    ax2 = fig.add_subplot(gs[0, 1])
    if kmin is not None:
        for m in sorted(kmin['m'].unique()):
            curve = kmin[kmin['m'] == m].sort_values('K')
            ax2.plot(curve['K'], curve[loss], lw=lw, color=palette_dict[m],
                     marker='o', markersize=10)
        ax2.set_xscale('log', base=2)
        ax2.set_yscale('log')
    else:
        ax2.text(0.5, 0.5, 'no data', ha='center', va='center',
                 transform=ax2.transAxes, fontsize=font_size, color='gray')
    ax2.set_title(r'Modes ($\gamma=0.5$)')
    ax2.set_xlabel(r'$K$')
    ax2.set_ylabel('')
    ax2.grid(True, alpha=0.3)

    # Shared discrete legend for the m values.
    ax_legend = fig.add_subplot(gs[1, :])
    ax_legend.axis('off')
    legend_elements = [Patch(facecolor=palette_dict[m], label=str(int(m)))
                       for m in m_values]
    legend = ax_legend.legend(handles=legend_elements, title=r'$m$',
                              loc='center', ncol=len(m_values), frameon=False,
                              fontsize=font_size)
    legend.get_title().set_fontsize(font_size)

    output_path = output_dir / f'{model_name}_prony.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f"[{model_name}] saved {output_path}\n")


if __name__ == '__main__':
    for model_name, m_key in MODEL_CONFIGS.items():
        build_model_figure(model_name, m_key)
