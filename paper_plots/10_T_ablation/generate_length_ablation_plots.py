"""Sequence-length (T) ablation plots.

For every model we reproduce the original 2x2 "combined" loss-vs-alpha figure
(see paper_plots/1_<MODEL>/generate_combined_plot.py), but instead of a single
sequence length we stack the different T values used in
scripts/Dataset_length_ablation/<MODEL>.sh.

Layout of each per-model figure:
    rows    -> sequence length T, including the original length used in the
               paper (e.g. 512, 1024 [original], 2048)
    columns -> dataset (Exponential, Polynomial, Impulse, Airy)
    hue     -> model size m (consistent Accent colors across all panels)

Reading *down* a column shows how the sequence length affects that dataset's
loss-vs-alpha curve, which is the whole point of the ablation.
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

# Datasets shown as columns: (file token, display title)
DATASETS = [
    ('exp', 'Exponential'),
    ('poly', 'Polynomial'),
    ('impulse', 'Impulse'),
    ('airy', 'Airy'),
]

# Per-model configuration. m_key / m_min / m_max mirror the original
# generate_combined_plot.py of each model. `T_values` come from the matching
# scripts/Dataset_length_ablation/<MODEL>.sh, and `orig_T` is the sequence
# length used by the original paper figure (the `_experiment_` result files).
# `airy_loss_max` reproduces the extra outlier filter that the S4 combined plot
# applies to the Airy panel.
MODEL_CONFIGS = {
    'LSTM': dict(m_key=MODEL_KEY_REGISTRY['LSTM'], m_min=4, m_max=256,
                 T_values=[512, 2048], orig_T=1024, airy_loss_max=None),
    'S4': dict(m_key=MODEL_KEY_REGISTRY['S4'], m_min=1, m_max=128,
               T_values=[512, 2048], orig_T=1024, airy_loss_max=1e-5),
    'TCN': dict(m_key=MODEL_KEY_REGISTRY['TCN'], m_min=4, m_max=256,
                T_values=[512, 2048], orig_T=1024, airy_loss_max=None),
    'Transformer': dict(m_key=MODEL_KEY_REGISTRY['Trans_hid_dim'], m_min=0,
                        m_max=1024, T_values=[64, 256], orig_T=128,
                        airy_loss_max=None),
}

results_dir = Path("results")
output_dir = Path("paper_plots/10_T_ablation")


def load_panel(model_name, dataset, T, cfg):
    """Return the per-(m, alpha) minimum-loss frame for one panel, or None.

    The original sequence length (cfg['orig_T']) is read from the paper's
    `_experiment_` result file; every other length comes from the ablation
    `_T{T}_` file. None is returned when the result file is missing or has no
    usable loss column (e.g. the TCN ablation runs that errored out and only
    contain empty `loss_history`).
    """
    if T == cfg['orig_T']:
        path = results_dir / f"{model_name}_{dataset}_experiment_data_clean.pkl"
    else:
        path = results_dir / f"{model_name}_{dataset}_T{T}_data_clean.pkl"
    if not path.exists():
        print(f"  [skip] missing file: {path.name}")
        return None

    df = pd.read_pickle(path)
    if loss not in df.columns:
        print(f"  [skip] no '{loss}' column in {path.name}")
        return None

    df = df.copy()
    df['m'] = df[cfg['m_key']]
    df['alpha'] = df['data.rho_param.alpha'] / df['data.rho_param.alpha'].max()
    df_min = df.groupby(['m', 'alpha'], as_index=False)[loss].min()
    df_min = df_min[(df_min['m'] > cfg['m_min']) & (df_min['m'] < cfg['m_max'])]

    if dataset == 'airy' and cfg['airy_loss_max'] is not None:
        df_min = df_min[df_min[loss] < cfg['airy_loss_max']]

    if df_min.empty:
        print(f"  [skip] empty after filtering: {path.name}")
        return None
    return df_min


def build_model_figure(model_name, cfg):
    print(f"[{model_name}] building length-ablation figure...")

    # Rows are every ablation length plus the original length, sorted so the
    # original slots into the natural T progression.
    row_Ts = sorted(set(cfg['T_values']) | {cfg['orig_T']})

    # First pass: load every panel and collect the global set of m values so
    # the Accent color mapping is identical across all subplots and the legend.
    panels = {}
    m_values = set()
    for T in row_Ts:
        for dataset, _ in DATASETS:
            df_min = load_panel(model_name, dataset, T, cfg)
            panels[(T, dataset)] = df_min
            if df_min is not None:
                m_values.update(df_min['m'].unique())

    if not m_values:
        print(f"[{model_name}] no usable data found, skipping figure.\n")
        return

    m_values = sorted(m_values)
    accent_colors = plt.cm.Accent(np.linspace(0, 1, len(m_values)))
    palette_dict = dict(zip(m_values, accent_colors))

    n_rows = len(row_Ts)
    n_cols = len(DATASETS)

    fig = plt.figure(figsize=(12 * n_cols, 10 * n_rows + 2))
    gs = GridSpec(n_rows + 1, n_cols, figure=fig,
                  height_ratios=[1] * n_rows + [0.15],
                  hspace=0.4, wspace=0.3)

    for r, T in enumerate(row_Ts):
        for c, (dataset, title) in enumerate(DATASETS):
            ax = fig.add_subplot(gs[r, c])
            df_min = panels[(T, dataset)]

            if df_min is not None:
                sns.lineplot(data=df_min, x='alpha', y=loss, hue='m', ax=ax,
                             palette=palette_dict, lw=lw, legend=False)
                ax.set_yscale('log')
            else:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=font_size,
                        color='gray')

            ax.grid(True, alpha=0.3)

            # Column titles only on the top row.
            if r == 0:
                ax.set_title(title)
            # Row label (T=...) on the left-most column; flag the original.
            if c == 0:
                row_tag = f'$T={T}$ (orig)' if T == cfg['orig_T'] else f'$T={T}$'
                ax.set_ylabel(f'{row_tag}\nLoss')
            else:
                ax.set_ylabel('')
            # alpha label only on the bottom row.
            if r == n_rows - 1:
                ax.set_xlabel(r'$\alpha$')
            else:
                ax.set_xlabel('')

    # Shared discrete legend for the m values, spanning all columns.
    ax_legend = fig.add_subplot(gs[n_rows, :])
    ax_legend.axis('off')
    legend_elements = [Patch(facecolor=palette_dict[m], label=str(int(m)))
                       for m in m_values]
    legend = ax_legend.legend(handles=legend_elements, title=r'$m$',
                              loc='center', ncol=len(m_values), frameon=False,
                              fontsize=font_size)
    legend.get_title().set_fontsize(font_size)

    output_path = output_dir / f'{model_name}_length_ablation.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f"[{model_name}] saved {output_path}\n")


if __name__ == '__main__':
    for model_name, cfg in MODEL_CONFIGS.items():
        build_model_figure(model_name, cfg)
