import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import autoroot
from matplotlib.gridspec import GridSpec
from pathlib import Path
from paper_plots.plot_config import set_plot_style
from paper_plots.model_registry import MODEL_KEY_REGISTRY

# --- Configuration ---
font_size = 40
figsize = (12, 8)
lw = 4
loss = 'train/rLoss_epoch.min'
m_min = 4  # Minimum m value to include
m_max = 256  # Maximum m value to include
hidden_size_target = 32  # The hidden size to extract from single-layer LSTM

# Apply consistent styling
set_plot_style(font_family='Times New Roman', font_size=font_size, figsize=figsize, palette='tab20')

# Get model keys
multilayer_key = 'module.layers.LSTM.num_layers'
single_layer_hidden_key = 'module.layers.LSTM.hidden_size'

# Define paths (autoroot changes cwd to project root)
results_dir = Path("results")
output_dir = Path("paper_plots/2_Multilayer_LSTM")

# Use tab20 colormap for m values with discrete colors
# Each m value gets a distinct color from the tab20 palette

def load_and_combine_data(task):
    """Load both single-layer (m=32, layer=1) and multilayer data and combine them.

    This function combines:
    - Single-layer LSTM with hidden_size=32 (treated as layer=1)
    - Multilayer LSTM with hidden_size=32 and varying number of layers (2, 3, 4, etc.)

    This gives a complete picture of how performance changes with number of layers.
    """
    # Load multilayer data
    df_multi = pd.read_pickle(results_dir / f"Multilayer_LSTM_{task}_data_clean.pkl")
    df_multi['m'] = df_multi[multilayer_key]
    df_multi['alpha'] = df_multi['data.rho_param.alpha'] / df_multi['data.rho_param.alpha'].max()

    # Load single-layer LSTM data
    df_single = pd.read_pickle(results_dir / f"LSTM_{task}_experiment_data_clean.pkl")
    df_single['hidden_size'] = df_single[single_layer_hidden_key]
    df_single['alpha'] = df_single['data.rho_param.alpha'] / df_single['data.rho_param.alpha'].max()

    # Filter for m=32 from single-layer data
    df_single_m32 = df_single[df_single['hidden_size'] == hidden_size_target].copy()
    df_single_m32['m'] = 1  # This represents 1 layer

    # Select only necessary columns and combine
    cols_to_keep = ['m', 'alpha', loss]
    df_combined = pd.concat([
        df_single_m32[cols_to_keep],
        df_multi[cols_to_keep]
    ], ignore_index=True)

    # Get minimum loss for each (m, alpha) combination
    df_min = df_combined.groupby(['m', 'alpha'], as_index=False)[loss].min()

    return df_min

# Create 2x2 subplot figure with GridSpec for more control
fig = plt.figure(figsize=(24, 20))
gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.15], hspace=0.4, wspace=0.3)

# Plot 1: Exponential (top-left)
ax1 = fig.add_subplot(gs[0, 0])
df_min = load_and_combine_data('exp')

# Create discrete color palette for m values - each m gets a unique color
m_values = sorted(df_min['m'].unique())
tab20_colors = plt.cm.tab20(np.linspace(0, 1, len(m_values)))
palette_dict = dict(zip(m_values, tab20_colors))

sns.lineplot(data=df_min, x='alpha', y=loss, hue='m', ax=ax1,
             palette=palette_dict, lw=lw, legend=False)

ax1.set_ylabel('Loss')
ax1.set_xlabel('')
ax1.set_title('Exponential')
ax1.set_yscale('log')
ax1.set_ylim(1.1*1e-9, 0.8*1e-6)
ax1.grid(True, alpha=0.3)

# Plot 2: Polynomial (top-right)
ax2 = fig.add_subplot(gs[0, 1])
df_min = load_and_combine_data('poly')

sns.lineplot(data=df_min, x='alpha', y=loss, hue='m', ax=ax2,
             palette=palette_dict, lw=lw, legend=False)

ax2.set_xlabel(r'$\alpha$')
ax2.set_ylabel('')
ax2.set_title('Polynomial')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_label_coords(1.04, 0.04)

# Plot 3: Impulse (bottom-left)
ax3 = fig.add_subplot(gs[1, 0])
df_min = load_and_combine_data('impulse')

sns.lineplot(data=df_min, x='alpha', y=loss, hue='m', ax=ax3,
             palette=palette_dict, lw=lw, legend=False)

ax3.set_ylabel('Loss')
ax3.set_xlabel('')
ax3.set_title('Impulse')
ax3.set_yscale('log')
ax3.grid(True, alpha=0.3)

# Plot 4: Airy (bottom-right)
ax4 = fig.add_subplot(gs[1, 1])
df_min = load_and_combine_data('airy')

# Keep legend for this plot to extract handles
line_plot = sns.lineplot(data=df_min, x='alpha', y=loss, hue='m', ax=ax4,
                         palette=palette_dict, lw=lw, legend=False)

ax4.set_xlabel(r'$\alpha$')
ax4.set_ylabel('')
ax4.set_title('Airy')
ax4.set_yscale('log')
ax4.grid(True, alpha=0.3)
ax4.xaxis.set_label_coords(1.04, 0.04)

# Add legend at the bottom spanning both columns
ax_legend = fig.add_subplot(gs[2, :])
ax_legend.axis('off')

# m_values is already defined from the first plot
# No need to create a temporary plot since we have m_values

# Create discrete legend with each m value
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=palette_dict[m], label=str(int(m))) for m in m_values]
legend = ax_legend.legend(handles=legend_elements, title='Number of Layers', loc='center',
                          ncol=len(m_values), frameon=False, fontsize=font_size)
legend.get_title().set_fontsize(font_size)

# Save the combined figure to the correct directory
output_path = output_dir / 'Multilayer_LSTM_combined.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
print(f"Saved {output_path}")
