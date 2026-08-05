import numpy as np
import pandas as pd
import autoroot
from pathlib import Path
from paper_plots.model_registry import MODEL_KEY_REGISTRY

# --- Configuration ---
MODEL_NAME = 'Multilayer_S4'
loss = 'train/rLoss_epoch.min'
m_min = 4  # Minimum m value to include
m_max = 256  # Maximum m value to include

# Get model key - for Multilayer_S4, we use n_layers
m_key = 'module.layers.Layer2.n_layers'

# Define paths (autoroot changes cwd to project root)
results_dir = Path("results")

# Calculate average loss increment at largest alpha compared to smallest alpha
# averaged over different m values, for each task type

tasks = ['exp', 'poly', 'impulse', 'airy']
results = {}

for task in tasks:
    # Load data
    df = pd.read_pickle(results_dir / f"{MODEL_NAME}_{task}_data_clean.pkl")
    df['m'] = df[m_key]
    df['alpha'] = df['data.rho_param.alpha']/df['data.rho_param.alpha'].max()

    # Get minimum loss for each (m, alpha) combination
    df_min = df.groupby(['m', 'alpha'], as_index=False)[loss].min()

    # Filter based on 'm' values (optional - may be commented out for Multilayer)
    # df_min = df_min[(df_min['m'] > m_min) & (df_min['m'] < m_max)]

    # Find the smallest and largest alpha values
    alpha_min = df_min['alpha'].min()
    alpha_max = df_min['alpha'].max()

    # Get losses at smallest and largest alpha
    loss_alpha_min = df_min[df_min['alpha'] == alpha_min].groupby('m')[loss].min()
    loss_alpha_max = df_min[df_min['alpha'] == alpha_max].groupby('m')[loss].min()

    # Calculate increment for each m
    # Using inner join to ensure we only compare when both alpha values exist
    increments = loss_alpha_max - loss_alpha_min

    # Calculate average increment across all m values
    avg_increment = increments.mean()

    # Calculate relative increment (as ratio, not percentage)
    relative_increments = increments / loss_alpha_min

    # Geometric mean (better for log-scale data)
    geometric_mean = np.exp(np.log(np.abs(relative_increments)).mean()) * np.sign(relative_increments.mean())

    # Calculate standard deviation and standard error
    std_dev = relative_increments.std()
    std_error = relative_increments.sem()

    results[task] = {
        'avg_absolute_increment': avg_increment,
        'geometric_mean': geometric_mean,
        'std': std_dev,
        'sem': std_error,
        'num_m_values': len(increments),
        'alpha_min': alpha_min,
        'alpha_max': alpha_max
    }

    print(f"\n{task.upper()}:")
    print(f"  Alpha range: {alpha_min:.6f} to {alpha_max:.6f}")
    print(f"  Average absolute loss increment: {avg_increment:.6e}")
    print(f"  Relative increment - Geometric mean: {geometric_mean:.6e}")
    print(f"  Standard deviation: {std_dev:.6e}")
    print(f"  Standard error: {std_error:.6e}")
    print(f"  Number of m values used: {len(increments)}")

print("\n" + "="*60)
print("SUMMARY - Geometric Mean")
print("="*60)
for task in tasks:
    print(f"{task:8s}: {results[task]['geometric_mean']:.6e} ± {results[task]['sem']:.6e}")

# LaTeX formatted output
def format_latex_scientific(value, error=None):
    """Format value in scientific notation for LaTeX"""
    if value == 0:
        return "$0$"

    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10**exponent)

    if error is not None:
        error_exponent = int(np.floor(np.log10(abs(error))))
        error_mantissa = error / (10**error_exponent)

        # Use same exponent for both value and error
        if error_exponent != exponent:
            error_mantissa = error / (10**exponent)

        return f"${mantissa:.2f} \\pm {error_mantissa:.2f} \\times 10^{{{exponent}}}$"
    else:
        return f"${mantissa:.2f} \\times 10^{{{exponent}}}$"

# Generate complete LaTeX table
print("\n" + "="*60)
print("COMPLETE LATEX TABLE")
print("="*60)

latex_table = r"""
\begin{table}[h]
\centering
\caption{Alpha Sensitivity Analysis: Relative Loss Increment from Minimum to Maximum $\alpha$ (Geometric Mean $\pm$ SEM)}
\label{tab:alpha_sensitivity_multilayer_s4}
\begin{tabular}{lcccc}
\hline
Model & Exponential & Polynomial & Impulse & Airy \\
\hline
"""

# Add data row
latex_values = [format_latex_scientific(results[task]['geometric_mean'], results[task]['sem']) for task in tasks]
latex_table += f"{MODEL_NAME.replace('_', ' ')} & " + " & ".join(latex_values) + " \\\\\n"

latex_table += r"""\hline
\end{tabular}
\end{table}
"""

print(latex_table)

# Also save to file
output_file = Path(f"paper_plots/2_{MODEL_NAME}") / f"{MODEL_NAME}_alpha_sensitivity_table.tex"
with open(output_file, 'w') as f:
    f.write(latex_table)
print(f"\nLaTeX table saved to: {output_file}")
