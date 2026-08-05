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

# Loss thresholds for different memory functions
LOSS_THRESHOLDS = {
    'exp': 1e-5,
    'poly': 1e-5,
    'impulse': 1e-1,
    'airy': 1e-3
}

# Apply consistent styling
set_plot_style(font_family='Times New Roman', font_size=font_size, figsize=figsize, palette='Accent')

# Get model key
m_key = MODEL_KEY_REGISTRY['LSTM']

# Define paths (autoroot changes cwd to project root)
results_dir = Path("results")
output_dir = Path("paper_plots/7_Training_Speed")

def find_steps_to_threshold(loss_history, threshold):
    """Find the first step where loss drops below threshold.

    Returns:
        int or None: Step number if threshold reached, None otherwise
    """
    for step, loss in enumerate(loss_history):
        if loss < threshold:
            return step
    # If threshold never reached, return None (will be filtered out)
    return None

def load_and_process_data(task, threshold):
    """Load training speed data and process."""
    df = pd.read_pickle(results_dir / f"LSTM_{task}_step2_data_clean.pkl")
    df['m'] = df[m_key]
    df['alpha'] = df['data.rho_param.alpha'] / df['data.rho_param.alpha'].max()

    # Calculate steps to threshold for each run
    df['steps_to_threshold'] = df['loss_history'].apply(
        lambda hist: find_steps_to_threshold(hist, threshold)
    )

    # Filter out runs that never reached threshold (steps_to_threshold is None)
    df_reached = df[df['steps_to_threshold'].notna()].copy()

    # Group by (m, alpha) and take the minimum steps across seeds (fastest training)
    df_min = df_reached.groupby(['m', 'alpha'], as_index=False)['steps_to_threshold'].min()

    return df_min

# Create 2x2 subplot figure with GridSpec + legend row
fig = plt.figure(figsize=(24, 20))
gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.15], hspace=0.4, wspace=0.3)

# Get all unique m values across all tasks to create consistent color palette
all_m_values = set()
for task in ['exp', 'poly', 'impulse', 'airy']:
    df = load_and_process_data(task, LOSS_THRESHOLDS[task])
    all_m_values.update(df['m'].unique())
m_values = sorted(all_m_values)

# Create Accent color palette for m values
accent_colors = plt.cm.Accent(np.linspace(0, 1, len(m_values)))
palette = dict(zip(m_values, accent_colors))

# Plot 1: Exponential (top-left)
ax1 = fig.add_subplot(gs[0, 0])
df_plot = load_and_process_data('exp', LOSS_THRESHOLDS['exp'])
sns.lineplot(data=df_plot, x='alpha', y='steps_to_threshold', hue='m', ax=ax1,
             palette=palette, lw=lw, legend=False)
ax1.set_ylabel('Training Steps')
ax1.set_xlabel('')
ax1.set_title('Exponential')
ax1.grid(True, alpha=0.3)

# Plot 2: Polynomial (top-right)
ax2 = fig.add_subplot(gs[0, 1])
df_plot = load_and_process_data('poly', LOSS_THRESHOLDS['poly'])
sns.lineplot(data=df_plot, x='alpha', y='steps_to_threshold', hue='m', ax=ax2,
             palette=palette, lw=lw, legend=False)
ax2.set_xlabel(r'$\alpha$')
ax2.set_ylabel('')
ax2.set_title('Polynomial')
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_label_coords(1.04, 0.04)

# Plot 3: Impulse (bottom-left)
ax3 = fig.add_subplot(gs[1, 0])
df_plot = load_and_process_data('impulse', LOSS_THRESHOLDS['impulse'])
sns.lineplot(data=df_plot, x='alpha', y='steps_to_threshold', hue='m', ax=ax3,
             palette=palette, lw=lw, legend=False)
ax3.set_ylabel('Training Steps')
ax3.set_xlabel('')
ax3.set_title('Impulse')
ax3.grid(True, alpha=0.3)

# Plot 4: Airy (bottom-right)
ax4 = fig.add_subplot(gs[1, 1])
df_plot = load_and_process_data('airy', LOSS_THRESHOLDS['airy'])
sns.lineplot(data=df_plot, x='alpha', y='steps_to_threshold', hue='m', ax=ax4,
             palette=palette, lw=lw, legend=False)
ax4.set_xlabel(r'$\alpha$')
ax4.set_ylabel('')
ax4.set_title('Airy')
ax4.grid(True, alpha=0.3)
ax4.xaxis.set_label_coords(1.04, 0.04)

# Add legend at the bottom spanning both columns
from matplotlib.patches import Patch
ax_legend = fig.add_subplot(gs[2, :])
ax_legend.axis('off')

# Create discrete legend with each m value
legend_elements = [Patch(facecolor=palette[m], label=str(int(m))) for m in m_values]
legend = ax_legend.legend(handles=legend_elements, title=r'$m$', loc='center',
                          ncol=len(m_values), frameon=False, fontsize=font_size)
legend.get_title().set_fontsize(font_size)

# Save the combined figure
output_path = output_dir / 'LSTM_training_speed_combined.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
print(f"Saved {output_path}")
