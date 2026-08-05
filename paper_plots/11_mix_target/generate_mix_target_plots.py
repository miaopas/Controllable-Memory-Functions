"""Mixed-target (exp <-> impulse) plots.

The `mix_exp_impulse` dataset blends an exponential and an impulse memory
function: rho = (1 - beta) * rho_exp + beta * rho_impulse, where the swept
parameter `data.rho_param.alpha` plays the role of beta (mixture weight). So
beta = 0 is a pure exponential target and beta = 1 is a pure impulse target,
and every intermediate beta is a *different target*.

Following the main-text combined plots (paper_plots/1_<MODEL>), we produce one
figure per model showing the minimum loss versus the target (beta), with one
curve per model size m.
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
beta_key = 'data.rho_param.alpha'  # swept alpha == mixture weight beta

set_plot_style(font_family='Times New Roman', font_size=font_size,
               figsize=figsize, palette='tab20')

# Per-model size-parameter key (same registry the other plots use).
MODEL_CONFIGS = {
    'LSTM': MODEL_KEY_REGISTRY['LSTM'],
    'S4': MODEL_KEY_REGISTRY['S4'],
    'TCN': MODEL_KEY_REGISTRY['TCN'],
    'Transformer': MODEL_KEY_REGISTRY['Trans_hid_dim'],
}

results_dir = Path("results")
output_dir = Path("paper_plots/11_mix_target")


def build_model_figure(model_name, m_key):
    print(f"[{model_name}] building mix-target figure...")
    path = results_dir / f"{model_name}_mix_target_experiment_data_clean.pkl"
    df = pd.read_pickle(path)

    df = df.copy()
    df['m'] = df[m_key]
    df['beta'] = df[beta_key]
    # Minimum loss over seeds for each (m, beta).
    df_min = df.groupby(['m', 'beta'], as_index=False)[loss].min()

    # Discrete Accent color per model size, consistent with the other figures.
    m_values = sorted(df_min['m'].unique())
    accent_colors = plt.cm.Accent(np.linspace(0, 1, len(m_values)))
    palette_dict = dict(zip(m_values, accent_colors))

    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 0.12], hspace=0.55)

    ax = fig.add_subplot(gs[0, 0])
    sns.lineplot(data=df_min, x='beta', y=loss, hue='m', ax=ax,
                 palette=palette_dict, lw=lw, legend=False)
    ax.set_yscale('log')
    ax.set_xlabel(r'$\beta$  (exp $\rightarrow$ impulse)')
    ax.set_ylabel('Loss')
    ax.set_title(model_name)
    ax.grid(True, alpha=0.3)

    # Shared discrete legend for the m values.
    ax_legend = fig.add_subplot(gs[1, 0])
    ax_legend.axis('off')
    legend_elements = [Patch(facecolor=palette_dict[m], label=str(int(m)))
                       for m in m_values]
    legend = ax_legend.legend(handles=legend_elements, title=r'$m$',
                              loc='center', ncol=len(m_values), frameon=False,
                              fontsize=font_size)
    legend.get_title().set_fontsize(font_size)

    output_path = output_dir / f'{model_name}_mix_target.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f"[{model_name}] saved {output_path}\n")


def build_combined_figure():
    """Single 2x2 figure: one equal-sized panel per model.

    Each model has its own set of sizes m (e.g. Transformer uses 32..256), so
    every panel carries its own compact m legend rather than a shared one.
    """
    print("[combined] building 2x2 mix-target figure...")
    # plt.subplots gives a uniform grid -> all panels are exactly the same size.
    fig, axes = plt.subplots(2, 2, figsize=(24, 18))

    for ax, (model_name, m_key) in zip(axes.flat, MODEL_CONFIGS.items()):
        path = results_dir / f"{model_name}_mix_target_experiment_data_clean.pkl"
        df = pd.read_pickle(path)
        df = df.copy()
        df['m'] = df[m_key]
        df['beta'] = df[beta_key]
        df_min = df.groupby(['m', 'beta'], as_index=False)[loss].min()

        m_values = sorted(df_min['m'].unique())
        accent_colors = plt.cm.Accent(np.linspace(0, 1, len(m_values)))
        palette_dict = dict(zip(m_values, accent_colors))

        sns.lineplot(data=df_min, x='beta', y=loss, hue='m', ax=ax,
                     palette=palette_dict, lw=lw, legend=False)
        ax.set_yscale('log')
        ax.set_title(model_name)
        ax.set_xlabel(r'$\beta$  (exp $\rightarrow$ impulse)')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)

        # Per-panel legend (m sets differ between models).
        legend_elements = [Patch(facecolor=palette_dict[m], label=str(int(m)))
                           for m in m_values]
        leg = ax.legend(handles=legend_elements, title=r'$m$', loc='lower right',
                        ncol=2, frameon=True, fontsize=24)
        leg.get_title().set_fontsize(24)

    fig.tight_layout(pad=2.5)
    output_path = output_dir / 'mix_target_combined.pdf'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f"[combined] saved {output_path}\n")


if __name__ == '__main__':
    for model_name, m_key in MODEL_CONFIGS.items():
        build_model_figure(model_name, m_key)
    build_combined_figure()
