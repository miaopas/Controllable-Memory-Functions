import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import autoroot
from matplotlib import cm
from matplotlib.colors import Normalize
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
output_dir = Path("paper_plots/7_TCN_Complexity")

# --- Memory functions ---
def exp(s, alpha):
    return np.exp(-s / 10 / alpha)

def poly(s, alpha):
    return 1 / (1 + s) ** (1 / 3 / alpha)

def impulse(s, alpha):
    alpha_scaled = np.floor(alpha * (T // 2))
    THRESHOLD = 0.5
    return ((s > alpha_scaled - THRESHOLD) & (s < alpha_scaled + THRESHOLD)).astype(float)

def airy(s, alpha):
    ai, aip, bi, bip = special.airy(40 * (s / T - 0.5 * alpha))
    return ai

# --- Cumulative tail energy function C(rho, s) ---
def cumulative_function(rho_func, alpha):
    full_range = np.arange(0, T)
    rho_values = np.abs(rho_func(full_range, alpha)) ** 2
    # Apply permutation to make it monotone non-increasing
    pi_indices = np.argsort(-rho_values)  # Sort in descending order
    rho_sorted = rho_values[pi_indices]
    return np.array([np.sum(rho_sorted[s:]) for s in range(T)])

# --- Main plotting ---
memory_functions = {
    'Exponential': exp,
    'Polynomial': poly,
    'Impulse': impulse,
    'Airy': airy
}

s_values = np.arange(1, T + 1)
alpha_values = np.linspace(0.1, 1.0, 5)

# Create 2x2 subplot figure with GridSpec + colorbar row
fig = plt.figure(figsize=(24, 20))
gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.1], hspace=0.4, wspace=0.3)

# Color mapping setup
norm = Normalize(vmin=min(alpha_values), vmax=max(alpha_values))
cmap = cm.coolwarm
sm = cm.ScalarMappable(cmap=cmap, norm=norm)

axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
titles = ['Exponential', 'Polynomial', 'Impulse', 'Airy']

for ax, (name, func), title in zip(axes, memory_functions.items(), titles):
    # Special handling for impulse function
    if name == 'Impulse':
        alpha_vals_impulse = np.linspace(0, 1, 5)
        for alpha in alpha_vals_impulse:
            color = cmap(norm(alpha if alpha > 0.01 else 0.1))
            C_values = cumulative_function(func, alpha)
            ax.plot(s_values, C_values, color=color, linewidth=3)
        ax.set_xlim(0, 16)
    else:
        for alpha in alpha_values:
            color = cmap(norm(alpha))
            C_values = cumulative_function(func, alpha)
            ax.plot(s_values, C_values, color=color, linewidth=3)
        ax.set_xlim(0, 64)

    ax.set_xlabel(r'$s$')
    ax.set_ylabel(r'$C(\rho,s)$')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

# Add colorbar at the bottom spanning both columns
ax_cbar = fig.add_subplot(gs[2, :])
ax_cbar.axis('off')

# Create horizontal colorbar
cbar = plt.colorbar(sm, ax=ax_cbar, orientation='horizontal',
                    fraction=0.8, aspect=40, pad=0.05)
cbar.set_label(r'$\alpha$', fontsize=font_size)
cbar.set_ticks([0.1, 0.325, 0.55, 0.775, 1.0])
cbar.set_ticklabels(['0.1', '0.325', '0.55', '0.775', '1.0'])
cbar.ax.tick_params(labelsize=font_size-5)

# Save the combined figure
output_path = output_dir / 'TCN_complexity_combined.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
print(f"Saved {output_path}")
