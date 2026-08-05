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
lw = 3
loss = 'train/rLoss1_epoch.min'

# Apply consistent styling
set_plot_style(font_family='Times New Roman', font_size=font_size, figsize=figsize, palette='Accent')

# Get model key
m_key = MODEL_KEY_REGISTRY['LSTM']

# Define paths
results_dir = Path("results")
output_dir = Path("paper_plots/7_TCN_Init_Sensitivity")

def load_and_process_data(task):
    """Load LSTM Adam initialization sensitivity data."""
    df = pd.read_pickle(results_dir / f"LSTM_{task}_step3_data_clean.pkl")
    df['m'] = df[m_key]
    df['alpha'] = df['data.rho_param.alpha'] / df['data.rho_param.alpha'].max()

    # Get data for largest model size
    df_max_m = df[df['m'] == df['m'].max()].copy()

    return df_max_m

# Create 2x2 subplot figure with GridSpec
fig = plt.figure(figsize=(24, 20))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

# Get color palette for seeds - collect all unique seeds across all tasks
all_seeds = set()
for task in ['exp', 'poly', 'impulse', 'airy']:
    df = load_and_process_data(task)
    all_seeds.update(df['run_specs.seed'].unique())
seed_values = sorted(all_seeds)
accent_colors = plt.cm.Accent(np.linspace(0, 1, len(seed_values)))
palette = dict(zip(seed_values, accent_colors))

# Plot 1: Exponential (top-left)
ax1 = fig.add_subplot(gs[0, 0])
df_plot = load_and_process_data('exp')
sns.scatterplot(data=df_plot, x='alpha', y=loss, hue='run_specs.seed',
                palette=palette, ax=ax1, legend=False, s=200)
ax1.set_ylabel('Loss')
ax1.set_xlabel('')
ax1.set_title('Exponential')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

# Plot 2: Polynomial (top-right)
ax2 = fig.add_subplot(gs[0, 1])
df_plot = load_and_process_data('poly')
sns.scatterplot(data=df_plot, x='alpha', y=loss, hue='run_specs.seed',
                palette=palette, ax=ax2, legend=False, s=200)
ax2.set_xlabel(r'$\alpha$')
ax2.set_ylabel('')
ax2.set_title('Polynomial')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_label_coords(1.04, 0.04)

# Plot 3: Impulse (bottom-left)
ax3 = fig.add_subplot(gs[1, 0])
df_plot = load_and_process_data('impulse')
sns.scatterplot(data=df_plot, x='alpha', y=loss, hue='run_specs.seed',
                palette=palette, ax=ax3, legend=False, s=200)
ax3.set_ylabel('Loss')
ax3.set_xlabel('')
ax3.set_title('Impulse')
ax3.set_yscale('log')
ax3.grid(True, alpha=0.3)

# Plot 4: Airy (bottom-right)
ax4 = fig.add_subplot(gs[1, 1])
df_plot = load_and_process_data('airy')
sns.scatterplot(data=df_plot, x='alpha', y=loss, hue='run_specs.seed',
                palette=palette, ax=ax4, legend=False, s=200)
ax4.set_xlabel(r'$\alpha$')
ax4.set_ylabel('')
ax4.set_title('Airy')
ax4.set_yscale('log')
ax4.grid(True, alpha=0.3)
ax4.xaxis.set_label_coords(1.04, 0.04)

# Save the combined figure
output_path = output_dir / 'LSTM_adam_init_combined.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
print(f"Saved {output_path}")
