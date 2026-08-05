import matplotlib.pyplot as plt
import matplotlib as mpl
import  seaborn as sns

def set_plot_style(font_family='serif', font_size=14, legend_font_size=14, figsize=(8, 6), palette = 'muted'):
    """
    Apply consistent matplotlib style across plots.

    Args:
        font_family (str): Font family for all text.
        font_size (int): Base font size.
        figsize (tuple): Default figure size (width, height).
    """


    colors = sns.color_palette(palette)  # Fetch the palette from seaborn

    # Convert the color palette to a list of RGB tuples
    colors = [mpl.colors.to_hex(c) for c in colors]  # C

    # Use serif fonts with Times as preferred option
    if font_family == 'Times New Roman':
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'Times']

    mpl.rcParams.update({
        'font.family': font_family if font_family != 'Times New Roman' else 'serif',
        'font.size': font_size,
        'figure.figsize': figsize,
        'axes.titlesize': font_size + 2,
        'axes.labelsize': font_size,
        'legend.fontsize': font_size - 15,
        'legend.title_fontsize': legend_font_size,
        'xtick.labelsize': font_size ,
        'ytick.labelsize': font_size ,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.6,
        'lines.linewidth': 2,
        'lines.markersize': 6,
        'lines.markersize': 10, 
    })

    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)