import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import autoroot
from pathlib import Path
from paper_plots.plot_config import set_plot_style
from paper_plots.model_registry import MODEL_KEY_REGISTRY

# --- Configuration ---
font_size = 40
legend_font_size = 30
lw = 3
figsize = (12, 8)
loss = 'train/rLoss1_epoch.min'

# Apply consistent styling
set_plot_style(font_family='Times New Roman', font_size=font_size,
               figsize=figsize, palette='tab20', legend_font_size=legend_font_size)

# Get model keys
m_key_1 = MODEL_KEY_REGISTRY['Trans_hid_dim']
m_key_2 = MODEL_KEY_REGISTRY['Trans_nhead']

# Define paths
results_dir = Path("results")
output_dir = Path("paper_plots/7_Transformer_Analysis")

# Load data
df = pd.read_pickle(results_dir / "RPETransformer_impulse_step4_data_clean.pkl")
df['m_1'] = df[m_key_1]
df['m_2'] = df[m_key_2]
df['style'] = df['module.layers.Layer1.style']
df['alpha'] = df['data.rho_param.alpha'] / df['data.rho_param.alpha'].max()
df_min = df.groupby(['style', 'alpha'], as_index=False)[loss].min()

# Create single plot
fig, ax1 = plt.subplots(figsize=figsize)

# Plot with different colors for each encoding style
palette = sns.color_palette('tab20', n_colors=df_min['style'].nunique())
sns.lineplot(data=df_min, x='alpha', y=loss, ax=ax1,
             palette=palette, hue='style', linewidth=lw)

ax1.set_xlabel(r'$\alpha$')
ax1.set_yscale('log')
ax1.set_ylabel('Loss')
ax1.legend(title='Position Encoding', bbox_to_anchor=(1.05, 0.8),
           loc='upper left', fontsize=legend_font_size)

# Save figure
output_path = output_dir / 'Transformer_position_encoding.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
print(f"Saved {output_path}")
