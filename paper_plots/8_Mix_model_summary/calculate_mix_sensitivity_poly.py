import numpy as np
import pandas as pd
import autoroot
from pathlib import Path

# --- Configuration ---
MEMORY_FUNCTION = 'poly'
loss = 'train/rLoss_epoch.min'
m_min = None  # Minimum m value to include (None = no minimum)
m_max = None  # Maximum m value to include (None = no maximum)

# Color configuration for LaTeX tables
BASELINE_MODE = 'average'  # Options: 'row', 'average', 'column'
                       # 'row': compare to row diagonal (e.g., LSTM+S4 vs LSTM+LSTM)
                       # 'average': compare to average of both diagonals
                       # 'column': compare to column diagonal (e.g., LSTM+S4 vs S4+S4)
COLOR_SCALE_MAX = 0.5  # Maximum percentage difference for color saturation (0.5 = 50%)

# Define model names
models = ['LSTM', 'S4', 'TCN', 'Transformer']

# Define paths (autoroot changes cwd to project root)
results_dir = Path("results")

# Calculate alpha sensitivity for each mixed model combination
# The sensitivity is measured as the geometric mean of relative loss increments
# from minimum to maximum alpha values

print(f"Calculating alpha sensitivity for {MEMORY_FUNCTION.upper()} memory function")
print(f"m range: {m_min if m_min is not None else 'all'} to {m_max if m_max is not None else 'all'}")
print("="*80)

# Initialize 4x4 sensitivity matrix and error matrix
sensitivity_matrix = np.zeros((len(models), len(models)))
error_matrix = np.zeros((len(models), len(models)))

# Loop through all model combinations
for i, model1 in enumerate(models):
    for j, model2 in enumerate(models):
        # Construct file path
        file_path = results_dir / f"Mixed_{model1}_{model2}_{MEMORY_FUNCTION}_experiment_data_clean.pkl"

        # Load data
        df = pd.read_pickle(file_path)

        # Convert loss column to numeric (handles string 'NaN' values)
        df[loss] = pd.to_numeric(df[loss], errors='coerce')

        # Normalize alpha
        df['alpha'] = df['data.rho_param.alpha'] / df['data.rho_param.alpha'].max()

        # Filter by m range if specified
        # Note: Mixed models may not have an 'm' parameter like multilayer models
        # If m column doesn't exist, we'll use all data
        if 'm' in df.columns:
            if m_min is not None:
                df = df[df['m'] >= m_min]
            if m_max is not None:
                df = df[df['m'] <= m_max]

        # Get minimum loss for each alpha value (across all runs)
        df_min = df.groupby('alpha', as_index=False)[loss].min()

        # Find the smallest and largest alpha values
        alpha_min = df_min['alpha'].min()
        alpha_max = df_min['alpha'].max()

        # Get losses at smallest and largest alpha for EACH run to calculate error
        # We need to track individual runs - use the original data grouped properly
        loss_at_alpha_min = df[df['alpha'] == alpha_min][loss].dropna().values
        loss_at_alpha_max = df[df['alpha'] == alpha_max][loss].dropna().values

        # Handle case where different numbers of samples exist
        # Use the minimum length to ensure we can calculate element-wise
        min_len = min(len(loss_at_alpha_min), len(loss_at_alpha_max))
        loss_at_alpha_min = loss_at_alpha_min[:min_len]
        loss_at_alpha_max = loss_at_alpha_max[:min_len]

        # Calculate relative increment for each run
        relative_increments = (loss_at_alpha_max - loss_at_alpha_min) / loss_at_alpha_min

        # Calculate mean and standard error
        relative_increment_mean = relative_increments.mean()
        relative_increment_sem = relative_increments.std() / np.sqrt(len(relative_increments))

        # Store in matrices
        sensitivity_matrix[i, j] = relative_increment_mean
        error_matrix[i, j] = relative_increment_sem

        print(f"{model1:12s} + {model2:12s}: {relative_increment_mean:.6e} ± {relative_increment_sem:.6e}")

print("\n" + "="*80)
print("SENSITIVITY MATRIX (4x4 Table)")
print("="*80)
print(f"{'':12s}", end='')
for model in models:
    print(f"{model:>15s}", end='')
print()
print("-"*80)
for i, model1 in enumerate(models):
    print(f"{model1:12s}", end='')
    for j, model2 in enumerate(models):
        print(f"{sensitivity_matrix[i, j]:15.6e}", end='')
    print()

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

def calculate_baseline(i, j, sensitivity_matrix):
    """Calculate baseline value for comparison based on BASELINE_MODE"""
    if BASELINE_MODE == 'row':
        return sensitivity_matrix[i, i]  # Row diagonal
    elif BASELINE_MODE == 'column':
        return sensitivity_matrix[j, j]  # Column diagonal
    elif BASELINE_MODE == 'average':
        return (sensitivity_matrix[i, i] + sensitivity_matrix[j, j]) / 2.0  # Average of both diagonals
    else:
        raise ValueError(f"Unknown BASELINE_MODE: {BASELINE_MODE}")

def calculate_cell_color(value, baseline):
    """Calculate LaTeX color command for a cell based on comparison to baseline

    Uses unsaturated colors for better readability across different scales:
    - Green!30!white = better (lower sensitivity = more robust to alpha changes)
    - Red!30!white = worse (higher sensitivity = less robust to alpha changes)
    """
    if baseline == 0 or np.isnan(value) or np.isnan(baseline):
        return ""

    # Calculate percentage difference
    pct_diff = (value - baseline) / baseline

    # Clamp to [-COLOR_SCALE_MAX, COLOR_SCALE_MAX]
    pct_diff = np.clip(pct_diff, -COLOR_SCALE_MAX, COLOR_SCALE_MAX)

    # Map to color intensity [0, 100]
    if pct_diff < 0:
        # Better (lower sensitivity) - unsaturated green
        intensity = int(abs(pct_diff) / COLOR_SCALE_MAX * 30)  # Max 30% saturation
        return f"\\cellcolor{{green!{intensity}!white}}"
    else:
        # Worse (higher sensitivity) - unsaturated red
        intensity = int(pct_diff / COLOR_SCALE_MAX * 30)  # Max 30% saturation
        return f"\\cellcolor{{red!{intensity}!white}}"

# Generate complete LaTeX table (non-colored version)
print("\n" + "="*80)
print("COMPLETE LATEX TABLE (NON-COLORED)")
print("="*80)

latex_table = r"""
\begin{table}[h]
\centering
\caption{Alpha Sensitivity for Mixed Models - """ + MEMORY_FUNCTION.upper() + r""" Memory Function: Relative Loss Increment from Minimum to Maximum $\alpha$ (Mean $\pm$ SEM)}
\label{tab:mixed_alpha_sensitivity_""" + MEMORY_FUNCTION + r"""}
\begin{tabular}{l""" + "c"*len(models) + r"""}
\hline
Model 1 $\downarrow$ / Model 2 $\rightarrow$"""

# Add column headers
for model in models:
    latex_table += f" & {model}"
latex_table += r""" \\
\hline
"""

# Add data rows
for i, model1 in enumerate(models):
    latex_table += model1
    for j, model2 in enumerate(models):
        latex_table += " & " + format_latex_scientific(sensitivity_matrix[i, j], error_matrix[i, j])
    latex_table += r" \\" + "\n"

latex_table += r"""\hline
\end{tabular}
\end{table}
"""

print(latex_table)

# Save to file
output_file = Path(f"paper_plots/8_Mix_model_summary") / f"mix_sensitivity_{MEMORY_FUNCTION}_table.tex"
with open(output_file, 'w') as f:
    f.write(latex_table)
print(f"\nLaTeX table saved to: {output_file}")

# Generate colored LaTeX table
print("\n" + "="*80)
print("COLORED LATEX TABLE")
print(f"Baseline mode: {BASELINE_MODE}")
print(f"Color scale: ±{COLOR_SCALE_MAX*100:.0f}%")
print("Using unsaturated colors (max 30% saturation)")
print("="*80)

colored_latex_table = r"""
% Requires \usepackage{xcolor} and \usepackage{colortbl} in preamble
% Color scheme: Green = better (lower sensitivity), Red = worse (higher sensitivity)
% Uses unsaturated colors for better readability
\begin{table}[h]
\centering
\caption{Alpha Sensitivity for Mixed Models - """ + MEMORY_FUNCTION.upper() + r""" Memory Function: Relative Loss Increment from Minimum to Maximum $\alpha$ (Mean $\pm$ SEM) - Colored}
\label{tab:mixed_alpha_sensitivity_""" + MEMORY_FUNCTION + r"""_colored}
\begin{tabular}{l""" + "c"*len(models) + r"""}
\hline
Model 1 $\downarrow$ / Model 2 $\rightarrow$"""

# Add column headers
for model in models:
    colored_latex_table += f" & {model}"
colored_latex_table += r""" \\
\hline
"""

# Add data rows with colors
for i, model1 in enumerate(models):
    colored_latex_table += model1
    for j, model2 in enumerate(models):
        # Calculate color
        if i == j:
            # Diagonal - light gray
            color_cmd = "\\cellcolor{gray!20}"
        else:
            # Off-diagonal - calculate color based on baseline
            baseline = calculate_baseline(i, j, sensitivity_matrix)
            color_cmd = calculate_cell_color(sensitivity_matrix[i, j], baseline)

        # Add colored cell
        colored_latex_table += " & " + color_cmd + format_latex_scientific(sensitivity_matrix[i, j], error_matrix[i, j])
    colored_latex_table += r" \\" + "\n"

colored_latex_table += r"""\hline
\end{tabular}
\end{table}
"""

print(colored_latex_table)

# Print color legend
print("\n" + "="*80)
print("COLOR LEGEND")
print("="*80)
print("Green (unsaturated): Better (lower sensitivity = more robust to alpha changes)")
print("Red (unsaturated): Worse (higher sensitivity = less robust to alpha changes)")
print("Gray: Diagonal (baseline)")
print(f"Baseline mode: {BASELINE_MODE}")
if BASELINE_MODE == 'row':
    print("  Each cell compared to its row diagonal (e.g., LSTM+S4 vs LSTM+LSTM)")
elif BASELINE_MODE == 'column':
    print("  Each cell compared to its column diagonal (e.g., LSTM+S4 vs S4+S4)")
elif BASELINE_MODE == 'average':
    print("  Each cell compared to average of both diagonals")

# Save colored version to file
output_file_colored = Path(f"paper_plots/8_Mix_model_summary") / f"mix_sensitivity_{MEMORY_FUNCTION}_table_colored.tex"
with open(output_file_colored, 'w') as f:
    f.write(colored_latex_table)
print(f"\nColored LaTeX table saved to: {output_file_colored}")
