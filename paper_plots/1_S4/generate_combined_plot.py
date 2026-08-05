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
MODEL_NAME = 'S4'
font_size = 40
figsize = (12, 8)
lw = 4
loss = 'train/rLoss_epoch.min'
m_min = 1 # Minimum m value to include
m_max = 128  # Maximum m value to include

# Apply consistent styling
set_plot_style(font_family='Times New Roman', font_size=font_size, figsize=figsize, palette='tab20')

# Get model key
m_key = MODEL_KEY_REGISTRY[MODEL_NAME]

# Define paths (autoroot changes cwd to project root)
results_dir = Path("results")
output_dir = Path(f"paper_plots/1_{MODEL_NAME}")

# Use Accent colormap for m values with discrete colors
# This is different from coolwarm used for alpha in memory function plots
# Each m value gets a distinct color from the Accent palette

# Create 2x2 subplot figure with GridSpec for more control
fig = plt.figure(figsize=(24, 20))
gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.15], hspace=0.4, wspace=0.3)

# Plot 1: Exponential (top-left)
ax1 = fig.add_subplot(gs[0, 0])
df = pd.read_pickle(results_dir / f"{MODEL_NAME}_exp_experiment_data_clean.pkl")
df['m'] = df[m_key]
df['alpha'] = df['data.rho_param.alpha']/df['data.rho_param.alpha'].max()
df_min = df.groupby(['m', 'alpha'], as_index=False)[loss].min()
df_min = df_min[(df_min['m'] > m_min) & (df_min['m'] < m_max)]

# Create discrete color palette for m values - each m gets a unique color
m_values = sorted(df_min['m'].unique())
accent_colors = plt.cm.Accent(np.linspace(0, 1, len(m_values)))
palette_dict = dict(zip(m_values, accent_colors))

sns.lineplot(data=df_min, x='alpha', y=loss, hue='m', ax=ax1,
             palette=palette_dict, lw=lw, legend=False)

ax1.set_ylabel('Loss')
ax1.set_xlabel('')
ax1.set_title('Exponential')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

# Plot 2: Polynomial (top-right)
ax2 = fig.add_subplot(gs[0, 1])
df = pd.read_pickle(results_dir / f"{MODEL_NAME}_poly_experiment_data_clean.pkl")
df['m'] = df[m_key]
df['alpha'] = df['data.rho_param.alpha']/df['data.rho_param.alpha'].max()
df_min = df.groupby(['m', 'alpha'], as_index=False)[loss].min()
df_min = df_min[(df_min['m'] > m_min) & (df_min['m'] < m_max)]

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
df = pd.read_pickle(results_dir / f"{MODEL_NAME}_impulse_experiment_data_clean.pkl")
df['m'] = df[m_key]
df['alpha'] = df['data.rho_param.alpha']/df['data.rho_param.alpha'].max()
df_min = df.groupby(['m', 'alpha'], as_index=False)[loss].min()
df_min = df_min[(df_min['m'] > m_min) & (df_min['m'] < m_max)]

sns.lineplot(data=df_min, x='alpha', y=loss, hue='m', ax=ax3,
             palette=palette_dict, lw=lw, legend=False)

ax3.set_ylabel('Loss')
ax3.set_xlabel('')
ax3.set_title('Impulse')
ax3.set_yscale('log')
ax3.grid(True, alpha=0.3)

# Plot 4: Airy (bottom-right)
ax4 = fig.add_subplot(gs[1, 1])
df = pd.read_pickle(results_dir / f"{MODEL_NAME}_airy_experiment_data_clean.pkl")
df['m'] = df[m_key]
df['alpha'] = df['data.rho_param.alpha']/df['data.rho_param.alpha'].max()
df_min = df.groupby(['m', 'alpha'], as_index=False)[loss].min()
df_min = df_min[(df_min['m'] > m_min) & (df_min['m'] < m_max)]
df_min = df_min[df_min[loss]<1e-5]

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

# Create discrete legend with each m value
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=palette_dict[m], label=str(int(m))) for m in m_values]
legend = ax_legend.legend(handles=legend_elements, title=r'$m$', loc='center',
                          ncol=len(m_values), frameon=False, fontsize=font_size)
legend.get_title().set_fontsize(font_size)

# Save the combined figure to the correct directory
output_path = output_dir / f'{MODEL_NAME}_combined.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
print(f"Saved {output_path}")
