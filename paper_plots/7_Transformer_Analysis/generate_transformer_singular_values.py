import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import autoroot
from matplotlib.gridspec import GridSpec
from pathlib import Path
from paper_plots.plot_config import set_plot_style
from scipy import special

# --- Configuration ---
font_size = 40
figsize = (12, 8)
T = 64

set_plot_style(font_family='Times New Roman', font_size=font_size, figsize=figsize, palette='coolwarm')

# Define paths
output_dir = Path("paper_plots/7_Transformer_Analysis")

# --- Memory functions ---
def exp(s, alpha):
    return np.exp(-s / alpha)

def poly(s, alpha):
    return 1 / (1 + s) ** (1 / alpha)

def impulse(s, alpha):
    alpha_scaled = np.floor(alpha * (T // 2))
    THRESHOLD = 0.5
    return ((s > alpha_scaled - THRESHOLD) & (s < alpha_scaled + THRESHOLD)).astype(float)

def airy(s, alpha):
    ai, aip, bi, bip = special.airy(40 * (s / T - 0.5 * alpha))
    return ai

# --- Generate attention matrix ---
def generate_attention_matrix(T, alpha, decay_fn):
    rho = decay_fn(np.arange(1, T), alpha)  # Start from s=1 to exclude diagonal
    A = np.zeros((T, T))
    for i in range(1, T):
        A[i, :i] = rho[i-1::-1]  # Fill with reversed prefix of rho
    return A

# --- Compute and collect singular values ---
def compute_singular_values_data(T, alpha_list, decay_fn):
    data = []
    for alpha in alpha_list:
        A = generate_attention_matrix(T, alpha, decay_fn)
        svals = np.linalg.svd(A, compute_uv=False)
        for i, s in enumerate(svals):
            data.append({'Index': i, 'Singular Value': s, 'Alpha': f'α = {alpha}'})
    return pd.DataFrame(data)

# --- Main plotting ---
memory_functions = {
    'Exponential': exp,
    'Polynomial': poly,
    'Impulse': impulse,
    'Airy': airy
}

alpha_list = [0.1, 0.4, 0.6, 1.0]

# Create 2x2 subplot figure with GridSpec
fig = plt.figure(figsize=(24, 20))
gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

# Create coolwarm palette for alpha values
alpha_colors = plt.cm.coolwarm(np.linspace(0, 1, len(alpha_list)))
alpha_palette = {f'α = {alpha}': color for alpha, color in zip(alpha_list, alpha_colors)}

axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
titles = ['Exponential', 'Polynomial', 'Impulse', 'Airy']

for ax, (name, func), title in zip(axes, memory_functions.items(), titles):
    df = compute_singular_values_data(T, alpha_list, func)
    sns.lineplot(data=df, x='Index', y='Singular Value', hue='Alpha',
                 ax=ax, linewidth=3, palette=alpha_palette)
    ax.set_xlabel('Index')
    ax.set_ylabel('Singular Value')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=font_size-10)

# Save the combined figure
output_path = output_dir / 'Transformer_singular_values_combined.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
print(f"Saved {output_path}")
