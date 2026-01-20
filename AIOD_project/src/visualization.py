"""
Visualization Module.

This module is designed to generate publication-quality figures for metabolomic data analysis.
It implements specialized plots for:
1. QC & Normalization Check (Sample Boxplots).
2. Data Distribution Analysis (Feature Boxplots).
3. Data Density & Transformation Assessment (Global Density + Gaussian Fit).
4. Feature Overview (Heatmap-like condensed boxplots).

Key design principles:
- Accessibility: Uses patterns and distinct markers for Black & White readability.
- Granularity: Overlays individual data points on boxplots.
- Robustness: Handles pagination for large datasets automatically.
"""

import math
import os

import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import norm

from src.config_visualization import *


def plot_boxplots(df, features, output_dir, file_prefix="boxplot", class_col='Class', ylabel='Peak Area',
                  features_per_page=6):
    """
    Generates Box Plots for selected features with individual data points overlaid.
    Useful for detailed univariate analysis of biomarkers.

    Args:
        df (pd.DataFrame): Dataset containing samples and class labels.
        features (list): List of feature names to plot.
        output_dir (str): Directory for saving plots.
        features_per_page (int): Number of subplots per PDF page.
    """
    os.makedirs(output_dir, exist_ok=True)

    unique_classes = df[class_col].unique()
    current_palette = DISCRETE_COLORS[:len(unique_classes)]
    class_marker_map = {cls: MARKERS[i % len(MARKERS)] for i, cls in enumerate(unique_classes)}

    num_features = len(features)
    num_pages = math.ceil(num_features / features_per_page)
    n_cols = 2
    n_rows = math.ceil(features_per_page / n_cols)

    print(f"Generating Box Plots: {num_features} features across {num_pages} file(s)...")

    for page in range(num_pages):
        start_idx = page * features_per_page
        end_idx = min(start_idx + features_per_page, num_features)
        current_features = features[start_idx:end_idx]

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 5 * n_rows))
        axes = axes.flatten()

        for i, feature in enumerate(current_features):
            ax = axes[i]

            # 1. Box Plot (Statistical Summary)
            sns.boxplot(x=class_col, y=feature, data=df, ax=ax,
                        palette=current_palette, showfliers=False,
                        width=0.5, linewidth=1.5)

            # 2. Strip Plot (Individual Points)
            sns.stripplot(x=class_col, y=feature, data=df, ax=ax,
                          hue=class_col, palette=current_palette,
                          dodge=False, jitter=True, size=6, alpha=0.7,
                          edgecolor='gray', linewidth=0.5, marker='o')

            ax.set_title(f"{feature} Distribution", fontweight='bold')
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Group")
            ax.grid(True, linestyle='--', alpha=0.3)

            if ax.get_legend():
                ax.get_legend().remove()

        # Hide unused axes
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Global Legend
        legend_elements = [
            Line2D([0], [0], marker=class_marker_map[cls], color='w', label=cls,
                   markerfacecolor=DISCRETE_COLORS[idx % len(DISCRETE_COLORS)],
                   markersize=10, markeredgecolor='k')
            for idx, cls in enumerate(unique_classes)
        ]
        fig.legend(handles=legend_elements, loc='upper right', title="Condition", bbox_to_anchor=(0.98, 0.98))

        plt.tight_layout(rect=[0, 0, 0.9, 0.95])

        filename = f"{file_prefix}_page{page + 1}.{SAVE_FORMAT}"
        plt.savefig(os.path.join(output_dir, filename), format=SAVE_FORMAT, bbox_inches='tight')
        plt.close(fig)


def plot_sample_distributions(df, output_dir, file_name="samples_qc_boxplot", class_col='Class', samples_per_page=20,
                              plot_title=None, show_sample_names=False, showfliers=False):
    """
    Generates Box Plots for QC purposes, visualizing the intensity distribution of EACH sample.
    Crucial for identifying technical outliers or normalization issues.

    Args:
        samples_per_page (int): Number of samples (X-axis) per plot to avoid overcrowding.
    """
    os.makedirs(output_dir, exist_ok=True)
    if plot_title is None: plot_title = "Sample Intensity Distributions"

    numeric_df = df.drop(columns=[class_col])
    classes = df[class_col]
    unique_classes = df[class_col].unique()
    current_palette = DISCRETE_COLORS[:len(unique_classes)]

    num_samples = len(df)
    num_pages = math.ceil(num_samples / samples_per_page)
    sample_ids = df.index.tolist()

    print(f"Generating Sample QC Boxplots: {num_samples} samples across {num_pages} file(s)...")

    for page in range(num_pages):
        start_idx = page * samples_per_page
        end_idx = min(start_idx + samples_per_page, num_samples)
        current_batch_ids = sample_ids[start_idx:end_idx]

        # Prepare subset for melting (Wide -> Long)
        batch_numeric = numeric_df.loc[current_batch_ids]
        batch_classes = classes.loc[current_batch_ids]

        batch_combined = batch_numeric.copy()
        batch_combined['__Class__'] = batch_classes
        batch_combined['__SampleID__'] = batch_combined.index

        df_melted = batch_combined.melt(id_vars=['__SampleID__', '__Class__'],
                                        var_name='Metabolite', value_name='Intensity')

        fig, ax = plt.subplots(figsize=(12, 4.8))

        sns.boxplot(data=df_melted, x='__SampleID__', y='Intensity',
                    hue='__Class__', hue_order=unique_classes,
                    palette=current_palette, showfliers=showfliers,
                    linewidth=0.5, width=0.8,
                    medianprops={'color': 'white', 'linewidth': 1.0,
                                 'path_effects': [pe.withStroke(linewidth=2.0, foreground='black')]},
                    ax=ax)

        # Plot Settings
        page_info = f" (Plot {page + 1}/{num_pages})" if num_pages > 1 else ""
        ax.set_title(f"{plot_title}{page_info}", fontweight='bold')
        ax.set_ylabel("Intensity Distribution (All Metabolites)")

        if show_sample_names:
            ax.set_xticks(range(len(current_batch_ids)))
            ax.set_xticklabels(current_batch_ids, rotation=90, fontsize=8)
            ax.set_xlabel("Sample Name")
        else:
            step = min(20, len(current_batch_ids)) if len(current_batch_ids) > 0 else 1
            tick_positions = range(0, len(current_batch_ids), step)
            tick_labels = range(start_idx, start_idx + len(current_batch_ids), step)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)
            ax.set_xlabel("Sample Index")

        # Legend
        handles, labels = ax.get_legend_handles_labels()
        median_proxy = Line2D([0], [0], color='white', linewidth=1.0,
                              path_effects=[pe.withStroke(linewidth=2.0, foreground='black')], label='Median')
        handles.append(median_proxy)
        labels.append("Median")
        ax.legend(handles=handles, labels=labels, title="Group", loc='best')
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{file_name}_page{page + 1}.{SAVE_FORMAT}"), format=SAVE_FORMAT,
                    bbox_inches='tight')
        plt.close(fig)


def plot_features_overview(df, output_dir, file_name="features_overview", class_col='Class', features_per_page=20,
                           plot_title=None):
    """
    Generates a dense overview of Feature Distributions.
    Plots multiple features on the same X-Axis to compare relative intensities side-by-side.
    """
    os.makedirs(output_dir, exist_ok=True)
    if plot_title is None: plot_title = "Metabolite Intensity Overview"

    all_features = [col for col in df.columns if col != class_col]
    num_features = len(all_features)
    num_pages = math.ceil(num_features / features_per_page)
    unique_classes = df[class_col].unique()
    current_palette = DISCRETE_COLORS[:len(unique_classes)]

    print(f"Generating Features Overview: {num_features} features across {num_pages} file(s)...")

    for page in range(num_pages):
        start_idx = page * features_per_page
        end_idx = min(start_idx + features_per_page, num_features)
        current_batch_features = all_features[start_idx:end_idx]

        # Subset & Melt
        subset_df = df[[class_col] + current_batch_features].copy()
        df_melted = subset_df.melt(id_vars=[class_col], value_vars=current_batch_features,
                                   var_name='Metabolite', value_name='Intensity')

        fig, ax = plt.subplots(figsize=(12, 4))

        sns.boxplot(data=df_melted, x='Metabolite', y='Intensity', hue=class_col,
                    palette=current_palette, showfliers=False,
                    linewidth=0.5, width=0.7,
                    medianprops={'color': 'white', 'linewidth': 1.0,
                                 'path_effects': [pe.withStroke(linewidth=2.0, foreground='black')]},
                    ax=ax)

        page_info = f" (Features {start_idx + 1}-{end_idx} of {num_features})" if num_pages > 1 else ""
        ax.set_title(f"{plot_title}{page_info}", fontweight='bold')
        ax.set_ylabel("Intensity")

        step = 20
        tick_positions = range(0, len(current_batch_features), step)
        tick_labels = range(start_idx, start_idx + len(current_batch_features), step)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel("Metabolite Index")
        ax.tick_params(axis='x', rotation=0)

        # Legend
        handles, labels = ax.get_legend_handles_labels()
        median_proxy = Line2D([0], [0], color='white', linewidth=1.0,
                              path_effects=[pe.withStroke(linewidth=2.0, foreground='black')], label='Median')
        handles.append(median_proxy)
        labels.append("Median")
        ax.legend(handles=handles, labels=labels, title="Group", loc='best')
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{file_name}_page{page + 1}.{SAVE_FORMAT}"), format=SAVE_FORMAT,
                    bbox_inches='tight')
        plt.close(fig)


def plot_global_density(df, output_dir, file_name="global_density_plot", class_col='Class', xlabel='Intensity',
                        add_gaussian=True, plot_title=None):
    """
    Generates a Global Density Plot (KDE + Histogram) pooling all features.
    Used to assess if data transformation (e.g., Log10) achieved Gaussian normality.

    Args:
        add_gaussian (bool): If True, overlays the theoretical Normal distribution (dashed line).
    """
    os.makedirs(output_dir, exist_ok=True)
    if plot_title is None: plot_title = "Global Distribution Analysis"

    numeric_df = df.copy()
    melted_df = numeric_df.melt(id_vars=[class_col], var_name='Feature', value_name='Intensity')
    melted_df = melted_df.dropna(subset=['Intensity'])

    n_classes = df[class_col].nunique()
    current_palette = DISCRETE_COLORS[:n_classes]
    unique_classes = df[class_col].unique()
    color_map = dict(zip(unique_classes, current_palette))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Empirical Density (KDE)
    sns.histplot(data=melted_df, x='Intensity', hue=class_col,
                 palette=current_palette, kde=True, element="step",
                 stat="density", common_norm=False, alpha=0.2, linewidth=1.0, ax=ax)

    # Overlay Theoretical Gaussian
    if add_gaussian:
        x_min, x_max = ax.get_xlim()
        x_axis = np.linspace(x_min, x_max, 500)

        for cls in unique_classes:
            cls_data = melted_df[melted_df[class_col] == cls]['Intensity']
            if len(cls_data) > 1:
                mu, std = norm.fit(cls_data)
                p = norm.pdf(x_axis, mu, std)
                ax.plot(x_axis, p, linestyle='--', linewidth=2.0, color=color_map[cls], alpha=0.9,
                        label=f'{cls} Theoretical Normal')

    ax.set_title(f"{plot_title}", fontweight='bold')
    ax.set_xlabel(f"{xlabel} (All Features Pooled)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.5)

    legend_handles = [mpatches.Patch(color=color_map[cls], label=cls) for cls in unique_classes]
    legend_handles.append(Line2D([0], [0], color='black', linestyle='-', linewidth=1.5, label='Empirical (KDE)'))
    if add_gaussian:
        legend_handles.append(Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='Gaussian/Normal'))
    ax.legend(handles=legend_handles, loc='best', title="Legend")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_name}.{SAVE_FORMAT}"), format=SAVE_FORMAT, bbox_inches='tight')
    plt.close(fig)