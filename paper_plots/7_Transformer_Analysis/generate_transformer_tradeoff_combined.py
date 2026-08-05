import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import autoroot
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from pathlib import Path
from paper_plots.plot_config import set_plot_style
from paper_plots.model_registry import MODEL_KEY_REGISTRY

# --- Configuration ---
font_size = 40
figsize = (12, 8)
lw = 3
loss = 'train/rLoss1_epoch.min'
m_fixed = 128  # Fixed total model dimension

# Apply consistent styling
set_plot_style(font_family='Times New Roman', font_size=font_size, figsize=figsize, palette='coolwarm')

# Get model keys
m_key_1 = MODEL_KEY_REGISTRY['Trans_hid_dim']
m_key_2 = MODEL_KEY_REGISTRY['Trans_nhead']

# Define paths
results_dir = Path("results")
output_dir = Path("paper_plots/7_Transformer_Analysis")

def load_and_process_data(task):
    """Load Transformer tradeoff data for a specific task."""
    # Handle both exp/poly and exponential/polynomial naming
    filename = f"Transformer_{task}_experiment_data_clean.pkl"
    df = pd.read_pickle(results_dir / filename)
    df['m_1'] = df[m_key_1]
    df['m_2'] = df[m_key_2]
    df['alpha'] = pd.Categorical(df['data.rho_param.alpha'] / df['data.rho_param.alpha'].max())
    df_min = df.groupby(['m_1', 'm_2', 'alpha'], as_index=False, observed=False)[loss].min()
    df_filtered = df_min[df_min['m_1'] == m_fixed]
    return df_filtered

# Create 2x2 subplot figure with GridSpec + colorbar row
fig = plt.figure(figsize=(24, 20))
gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.1], hspace=0.4, wspace=0.3)

# Create color map for categorical alpha
cmap = plt.cm.coolwarm

# Plot 1: Exponential (top-left)
ax1 = fig.add_subplot(gs[0, 0])
df_plot = load_and_process_data('exp')
for i, alpha_level in enumerate(df_plot['alpha'].cat.categories):
    subset = df_plot[df_plot['alpha'] == alpha_level]
    color = cmap(i / (len(df_plot['alpha'].cat.categories) - 1))
    sns.lineplot(data=subset, x='m_2', y=loss, ax=ax1,
                 color=color, linewidth=lw, label=None, legend=False)
ax1.set_xlabel('')
ax1.set_ylabel('Loss')
ax1.set_title('Exponential')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

# Plot 2: Polynomial (top-right)
ax2 = fig.add_subplot(gs[0, 1])
df_plot = load_and_process_data('poly')
for i, alpha_level in enumerate(df_plot['alpha'].cat.categories):
    subset = df_plot[df_plot['alpha'] == alpha_level]
    color = cmap(i / (len(df_plot['alpha'].cat.categories) - 1))
    sns.lineplot(data=subset, x='m_2', y=loss, ax=ax2,
                 color=color, linewidth=lw, label=None, legend=False)
ax2.set_xlabel(r'$n_h$')
ax2.set_ylabel('')
ax2.set_title('Polynomial')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_label_coords(1.04, 0.04)

# Plot 3: Impulse (bottom-left)
ax3 = fig.add_subplot(gs[1, 0])
df_plot = load_and_process_data('impulse')
# Filter out very small alpha values for impulse
df_plot = df_plot[df_plot['data.rho_param.alpha'] > 0.01] if 'data.rho_param.alpha' in df_plot.columns else df_plot
for i, alpha_level in enumerate(df_plot['alpha'].cat.categories):
    subset = df_plot[df_plot['alpha'] == alpha_level]
    color = cmap(i / (len(df_plot['alpha'].cat.categories) - 1))
    sns.lineplot(data=subset, x='m_2', y=loss, ax=ax3,
                 color=color, linewidth=lw, label=None, legend=False)
ax3.set_xlabel('')
ax3.set_ylabel('Loss')
ax3.set_title('Impulse')
ax3.set_yscale('log')
ax3.grid(True, alpha=0.3)

# Plot 4: Airy (bottom-right)
ax4 = fig.add_subplot(gs[1, 1])
df_plot = load_and_process_data('airy')
for i, alpha_level in enumerate(df_plot['alpha'].cat.categories):
    subset = df_plot[df_plot['alpha'] == alpha_level]
    color = cmap(i / (len(df_plot['alpha'].cat.categories) - 1))
    sns.lineplot(data=subset, x='m_2', y=loss, ax=ax4,
                 color=color, linewidth=lw, label=None, legend=False)
ax4.set_xlabel(r'$n_h$')
ax4.set_ylabel('')
ax4.set_title('Airy')
ax4.set_yscale('log')
ax4.grid(True, alpha=0.3)
ax4.xaxis.set_label_coords(1.04, 0.04)

# Add colorbar at the bottom
ax_cbar = fig.add_subplot(gs[2, :])
ax_cbar.axis('off')

# Create colorbar
norm = mcolors.Normalize(vmin=0, vmax=1)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = plt.colorbar(sm, ax=ax_cbar, orientation='horizontal',
                    fraction=0.8, aspect=40, pad=0.05)
cbar.set_label(r'$\alpha$', fontsize=font_size)
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
cbar.set_ticklabels(['0', '0.25', '0.5', '0.75', '1'])
cbar.ax.tick_params(labelsize=font_size-5)

# Save the combined figure
output_path = output_dir / 'Transformer_tradeoff_combined.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
print(f"Saved {output_path}")
