"""Result processing utilities for wandb data retrieval and PDF report generation."""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from matplotlib.backends.backend_pdf import PdfPages
from omegaconf import DictConfig
from pandas import json_normalize
from tqdm import tqdm

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

# Import from paper_plots to avoid circular dependencies
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "paper_plots"))
from model_registry import MODEL_KEY_REGISTRY


def process_results_after_training(cfg: DictConfig):
    """Process and log results after all Ray Tune trials complete.

    Args:
        cfg: A DictConfig configuration composed by Hydra.
    """
    # Extract wandb project and entity from config
    project_name = cfg.logger.wandb.project
    entity = cfg.logger.wandb.entity
    results_dir = "results"

    # Infer model_name from tags
    model_name = None
    tags = cfg.run_specs.tags
    registry_lower = {k.lower(): k for k in MODEL_KEY_REGISTRY.keys()}
    for tag in tags:
        tag_lower = str(tag).lower()
        if tag_lower in registry_lower:
            model_name = registry_lower[tag_lower]
            break

    # Step 1: Get and process data (always executed)
    df = get_project_data(project_name, entity=entity, results_dir=results_dir)

    if model_name is None:
        # Write error to log file
        import traceback

        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)

        error_log_path = results_path / f"{project_name}_error.log"

        with open(error_log_path, 'w') as f:
            f.write(f"Error processing results for project: {project_name}\n")
            f.write(f"Entity: {entity}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Error: Could not infer model_name from tags\n")
            f.write(f"Tags: {tags}\n")
            f.write(f"Available models in registry: {list(MODEL_KEY_REGISTRY.keys())}\n")
        return

    try:

        # Step 2: Generate PDF report
        pdf_path = generate_result_pdf(
            project_name=project_name,
            model_name=model_name,
            results_dir=results_dir,
            threshold=1e-6,
        )

    except Exception as e:
        # Write error to log file instead of generating PDF
        import traceback

        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)

        error_log_path = results_path / f"{project_name}_{model_name}_error.log"

        with open(error_log_path, 'w') as f:
            f.write(f"Error processing results for project: {project_name}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Entity: {entity}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Error: {str(e)}\n\n")
            f.write("Traceback:\n")
            f.write(traceback.format_exc())


def get_project_data(
    project_name: str,
    entity: str = "Long_Memory_Arena",
    results_dir: str = "results",
    save: bool = True,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Retrieve and process data from a wandb project.

    Args:
        project_name: Name of the wandb project
        entity: Wandb entity name
        results_dir: Directory to save the processed dataframe
        save: Whether to save the processed dataframe to disk
        force_refresh: If False and data file exists, load from cache instead of fetching

    Returns:
        Processed pandas DataFrame with experiment results
    """
    # Check if cached data exists
    results_path = Path(results_dir)
    output_file = results_path / f"{project_name}_data_clean.pkl"

    if not force_refresh and output_file.exists():
        print(f"Loading cached data from {output_file}")
        return pd.read_pickle(output_file)

    print(f"Retrieving data for {project_name}...")
    api = wandb.Api()

    # Get runs from the project
    runs = api.runs(f"{entity}/{project_name}")

    summary_list, config_list, name_list, tag_list = [], [], [], []
    loss_list = []

    for run in tqdm(runs, desc="Processing runs"):
        # Get summary metrics
        summary_list.append(run.summary._json_dict)

        # Get hyperparameters (excluding internal wandb params)
        config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

        # Get run metadata
        name_list.append(run.name)
        tag_list.append(run.tags)

        # Get loss history
        try:
            history = run.scan_history(keys=["train/rLoss1_epoch"])
            losses = [row["train/rLoss1_epoch"] for row in history if "train/rLoss1_epoch" in row]
        except Exception as e:
            print(f"Failed to get loss for run {run.name}: {e}")
            losses = []

        loss_list.append(losses)

    # Create DataFrame
    df = pd.DataFrame({
        "summary": summary_list,
        "config": config_list,
        "name": name_list,
        "tag": tag_list,
        "loss_history": loss_list,
    })

    # Process and flatten nested dictionaries
    df['config_dict'] = df['config'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    df['summary_dict'] = df['summary'].apply(lambda x: eval(x) if isinstance(x, str) else x)

    expanded_config = json_normalize(df['config_dict'])
    expanded_summary = json_normalize(df['summary_dict'])

    df = df.drop(columns=['config', 'summary'])
    df = pd.concat([df, expanded_config, expanded_summary], axis=1)

    if save:
        # Create results directory if it doesn't exist
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)

        # Save processed dataframe
        output_file = results_path / f"{project_name}_data_clean.pkl"
        df.to_pickle(output_file)
        print(f"Saved processed data to {output_file}")

    return df


def plot_loss_vs_m(
    df: pd.DataFrame,
    m_key: str,
    title: str = "Loss vs Model Size",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Plot minimum loss vs model size for different alpha values.

    Args:
        df: Processed dataframe with experiment results
        m_key: Column name for model size parameter
        title: Plot title
        figsize: Figure size tuple

    Returns:
        Matplotlib figure object
    """
    df_plot = df.copy()
    df_plot['m'] = df_plot[m_key]

    df_min = df_plot.groupby(['m', 'data.rho_param.alpha'], as_index=False)['train/rLoss1_epoch.min'].min()

    fig, ax = plt.subplots(figsize=figsize)
    palette = sns.color_palette("tab10", n_colors=len(df_min['data.rho_param.alpha'].unique()))

    sns.lineplot(
        data=df_min,
        x='m',
        y='train/rLoss1_epoch.min',
        hue='data.rho_param.alpha',
        marker='o',
        palette=palette,
        ax=ax,
    )

    ax.set_yscale('log')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Model Size (m)', fontsize=12)
    ax.set_ylabel('Minimum Training Loss (log scale)', fontsize=12)
    ax.legend(title='Alpha (ρ)', title_fontsize=11, fontsize=10)
    ax.grid(True, which="both", ls="--", alpha=0.6)

    plt.tight_layout()
    return fig


def plot_loss_vs_alpha(
    df: pd.DataFrame,
    m_key: str,
    title: str = "Loss vs Alpha",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Plot minimum loss vs alpha for different model sizes.

    Args:
        df: Processed dataframe with experiment results
        m_key: Column name for model size parameter
        title: Plot title
        figsize: Figure size tuple

    Returns:
        Matplotlib figure object
    """
    df_plot = df.copy()
    df_plot['m'] = df_plot[m_key]

    df_min = df_plot.groupby(['m', 'data.rho_param.alpha'], as_index=False)['train/rLoss1_epoch.min'].min()

    fig, ax = plt.subplots(figsize=figsize)
    palette = sns.color_palette("tab10", n_colors=len(df_min['m'].unique()))

    sns.lineplot(
        data=df_min,
        x='data.rho_param.alpha',
        y='train/rLoss1_epoch.min',
        hue='m',
        marker='o',
        palette=palette,
        ax=ax,
    )

    ax.set_yscale('log')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Alpha (ρ parameter)', fontsize=12)
    ax.set_ylabel('Minimum Training Loss (log scale)', fontsize=12)
    ax.legend(title='Model Size (m)', title_fontsize=11, fontsize=10, ncol=2)
    ax.grid(True, which="both", ls="--", alpha=0.6)

    plt.tight_layout()
    return fig


def plot_steps_to_threshold(
    df: pd.DataFrame,
    m_key: str,
    threshold: float = 1e-6,
    title: str = "Steps to Reach Threshold",
    figsize: tuple = (10, 6),
) -> Optional[plt.Figure]:
    """
    Plot minimum steps to reach loss threshold vs alpha for different model sizes.

    Args:
        df: Processed dataframe with experiment results
        m_key: Column name for model size parameter
        threshold: Loss threshold value
        title: Plot title
        figsize: Figure size tuple

    Returns:
        Matplotlib figure object or None if no data reaches threshold
    """
    df_plot = df.copy()
    df_plot['m'] = df_plot[m_key]

    # Calculate steps to reach threshold
    df_plot['steps_to_reach_e'] = df_plot['loss_history'].apply(
        lambda losses: next((i for i, val in enumerate(losses) if float(val) < float(threshold)), None)
    )

    # Drop NAs
    plot_df = df_plot.dropna(subset=['steps_to_reach_e'])

    if len(plot_df) == 0:
        print(f"No runs reached threshold {threshold}")
        return None

    # Group by alpha and m, and get the min steps
    plot_df_min = plot_df.groupby(['data.rho_param.alpha', 'm'], as_index=False)['steps_to_reach_e'].min()

    fig, ax = plt.subplots(figsize=figsize)
    palette = sns.color_palette("tab10", n_colors=plot_df_min['m'].nunique())

    sns.lineplot(
        data=plot_df_min,
        x='data.rho_param.alpha',
        y='steps_to_reach_e',
        hue='m',
        marker='o',
        palette=palette,
        ax=ax,
    )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Alpha (ρ parameter)', fontsize=12)
    ax.set_ylabel(f'Steps to Reach Loss < {threshold}', fontsize=12)
    ax.legend(title='Model Size (m)', title_fontsize=11, fontsize=10, ncol=2)
    ax.grid(True, alpha=0.6)

    plt.tight_layout()
    return fig


def generate_result_pdf(
    project_name: str,
    model_name: str,
    results_dir: str = "results",
    threshold: float = 1e-6,
    conditions: Optional[List[str]] = None,
    output_filename: Optional[str] = None,
    m_key_dict: Optional[Dict[str, str]] = None,
    overwrite: bool = True,
) -> str:
    """
    Generate a PDF report with plots for a given project.

    Args:
        project_name: Name of the wandb project
        model_name: Name of the model (key in m_key_dict)
        results_dir: Directory containing the processed dataframes
        threshold: Loss threshold for convergence analysis
        conditions: List of query conditions to filter the dataframe
        output_filename: Optional custom output filename
        m_key_dict: Optional custom dictionary mapping model names to their size parameter keys.
                   If None, uses MODEL_KEY_REGISTRY.
        overwrite: If False and PDF exists, skip generation and return existing path

    Returns:
        Path to the generated PDF file
    """
    # Use registry if no custom dict provided
    if m_key_dict is None:
        m_key_dict = MODEL_KEY_REGISTRY

    # Load processed data
    results_path = Path(results_dir)
    data_file = results_path / f"{project_name}_data_clean.pkl"

    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    df = pd.read_pickle(data_file)

    # Get model size key
    if model_name not in m_key_dict:
        raise ValueError(f"Model {model_name} not found in m_key_dict. Available models: {list(m_key_dict.keys())}")

    m_key = m_key_dict[model_name]

    # Apply conditions if provided
    if conditions:
        for condition in conditions:
            df = df.query(condition)

    # Generate output filename
    if output_filename is None:
        output_filename = f"{project_name}_{model_name}_report.pdf"

    output_path = results_path / output_filename

    # Check if PDF already exists
    if not overwrite and output_path.exists():
        print(f"PDF already exists at {output_path}, skipping generation")
        return str(output_path)

    # Create PDF with plots
    with PdfPages(output_path) as pdf:
        # Plot 1: Loss vs Model Size
        fig1 = plot_loss_vs_m(
            df, m_key,
            title=f"{project_name}: Loss vs Model Size"
        )
        pdf.savefig(fig1, bbox_inches='tight')
        plt.close(fig1)

        # Plot 2: Loss vs Alpha
        fig2 = plot_loss_vs_alpha(
            df, m_key,
            title=f"{project_name}: Loss vs Alpha"
        )
        pdf.savefig(fig2, bbox_inches='tight')
        plt.close(fig2)

        # Plot 3: Steps to Threshold
        fig3 = plot_steps_to_threshold(
            df, m_key, threshold,
            title=f"{project_name}: Convergence Speed (threshold={threshold})"
        )
        if fig3 is not None:
            pdf.savefig(fig3, bbox_inches='tight')
            plt.close(fig3)

    print(f"PDF report saved to {output_path}")
    return str(output_path)


if __name__ == "__main__":
    # Example usage
    project = "LSTM_exp_step3"
    df = get_project_data(project, results_dir="results")
    pdf_path = generate_result_pdf(
        project,
        "LSTM",
        results_dir="results",
        threshold=1e-7
    )
