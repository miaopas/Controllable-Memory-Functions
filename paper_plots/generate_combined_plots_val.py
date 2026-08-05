"""Validation-loss versions of the main-text combined plots.

Mirrors paper_plots/1_<MODEL>/generate_combined_plot.py exactly (2x2 grid over
the four targets, one curve per model size m, Accent palette, log-y, shared
bottom legend) but plots validation loss instead of training loss.

There is no `val/rLoss_epoch.min` summary (only train metrics get a min
aggregate via define_metric), so we use the final-epoch `val/rLoss_epoch`; the
per-(m, alpha) groupby-min then reduces over seeds, matching the train scripts.

The train scripts' magnitude-specific tweaks (LSTM's hardcoded exp ylim, S4's
airy <1e-5 outlier filter) are intentionally dropped here because they were
tuned to training-loss values and would clip/drop valid validation points;
panels autoscale instead.

Output: paper_plots/1_<MODEL>/<MODEL>_combined_val.pdf
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

# --- Configuration (matches the train combined plots) ---
font_size = 40
figsize = (12, 8)
lw = 4
loss = 'val/rLoss_epoch'  # validation analog of train/rLoss_epoch.min

set_plot_style(font_family='Times New Roman', font_size=font_size,
               figsize=figsize, palette='tab20')

# Datasets as the four panels: (file token, display title).
DATASETS = [
    ('exp', 'Exponential'),
    ('poly', 'Polynomial'),
    ('impulse', 'Impulse'),
    ('airy', 'Airy'),
]

# Per-model size key + inclusion range, copied from each 1_<MODEL> script.
MODEL_CONFIGS = {
    'LSTM': dict(m_key=MODEL_KEY_REGISTRY['LSTM'], m_min=4, m_max=256),
    'S4': dict(m_key=MODEL_KEY_REGISTRY['S4'], m_min=1, m_max=128),
    'TCN': dict(m_key=MODEL_KEY_REGISTRY['TCN'], m_min=4, m_max=256),
    'Transformer': dict(m_key=MODEL_KEY_REGISTRY['Trans_hid_dim'], m_min=0,
                        m_max=1024),
}

results_dir = Path("results")


def load_panel(model_name, dataset, cfg):
    """Per-(m, alpha) minimum validation loss over seeds for one panel."""
    path = results_dir / f"{model_name}_{dataset}_experiment_data_clean.pkl"
    df = pd.read_pickle(path)
    df = df.copy()
    df['m'] = df[cfg['m_key']]
    df['alpha'] = df['data.rho_param.alpha'] / df['data.rho_param.alpha'].max()
    df_min = df.groupby(['m', 'alpha'], as_index=False)[loss].min()
    df_min = df_min[(df_min['m'] > cfg['m_min']) & (df_min['m'] < cfg['m_max'])]
    return df_min


def build_model_figure(model_name, cfg):
    print(f"[{model_name}] building validation combined figure...")

    panels = {ds: load_panel(model_name, ds, cfg) for ds, _ in DATASETS}

    # One Accent color per model size, consistent across panels + legend.
    m_values = sorted(set().union(*[set(p['m'].unique()) for p in panels.values()]))
    accent_colors = plt.cm.Accent(np.linspace(0, 1, len(m_values)))
    palette_dict = dict(zip(m_values, accent_colors))

    fig = plt.figure(figsize=(24, 20))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.15],
                  hspace=0.4, wspace=0.3)

    # (gridspec position, ylabel, xlabel) for each of the four panels.
    panel_layout = [
        (gs[0, 0], 'Loss', ''),
        (gs[0, 1], '', r'$\alpha$'),
        (gs[1, 0], 'Loss', ''),
        (gs[1, 1], '', r'$\alpha$'),
    ]

    for (pos, ylabel, xlabel), (dataset, title) in zip(panel_layout, DATASETS):
        ax = fig.add_subplot(pos)
        sns.lineplot(data=panels[dataset], x='alpha', y=loss, hue='m', ax=ax,
                     palette=palette_dict, lw=lw, legend=False)
        ax.set_title(title)
        ax.set_yscale('log')
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)
        if xlabel:  # right-column alpha label placement, as in the train plots
            ax.xaxis.set_label_coords(1.04, 0.04)

    # Shared discrete legend at the bottom spanning both columns.
    ax_legend = fig.add_subplot(gs[2, :])
    ax_legend.axis('off')
    legend_elements = [Patch(facecolor=palette_dict[m], label=str(int(m)))
                       for m in m_values]
    legend = ax_legend.legend(handles=legend_elements, title=r'$m$',
                              loc='center', ncol=len(m_values), frameon=False,
                              fontsize=font_size)
    legend.get_title().set_fontsize(font_size)

    output_path = Path(f"paper_plots/1_{model_name}") / f'{model_name}_combined_val.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f"[{model_name}] saved {output_path}\n")


if __name__ == '__main__':
    for model_name, cfg in MODEL_CONFIGS.items():
        build_model_figure(model_name, cfg)
