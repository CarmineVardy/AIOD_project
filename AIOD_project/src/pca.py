"""
Principal Component Analysis (PCA) Module.

This module provides a complete workflow for PCA, including:
1. Dimensionality Reduction (perform_pca).
2. Advanced Visualization (Scores, Loadings, Scree Plots, Loading Profiles).
3. Multivariate Anomaly Detection (Hotelling's T2 vs Q-Residuals).
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import f, chi2
from src.config_visualization import *


def perform_pca(df, n_components=None, scaling='autoscaling'):
    """
    Performs Principal Component Analysis (PCA) on the dataset.

    Preprocessing:
    - Automatically handles scaling (Autoscaling or Pareto) as PCA is scale-dependent.

    Args:
        df (pd.DataFrame): Input dataframe (Samples x Features).
        n_components (int): Number of components to keep.
        scaling (str): 'autoscaling', 'pareto', or None.

    Returns:
        dict: Contains the trained model, scores, loadings, and variance explained.
    """
    # Separate numeric data
    df_numeric = df.drop(columns=['Class'])
    data_mat = df_numeric.values

    # 1. Scaling
    if scaling == 'autoscaling':
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_mat)
    else:
        data_scaled = data_mat

    # 2. PCA Calculation
    pca = PCA(n_components=n_components)
    scores_data = pca.fit_transform(data_scaled)

    # 3. Formatting
    pc_labels = [f"PC{i + 1}" for i in range(scores_data.shape[1])]
    scores_df = pd.DataFrame(data=scores_data, index=df.index, columns=pc_labels)
    loadings_df = pd.DataFrame(data=pca.components_.T, index=df_numeric.columns, columns=pc_labels)

    return {
        'model': pca,
        'scores': scores_df,
        'loadings': loadings_df,
        'explained_variance': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
        'data_scaled': data_scaled
    }


def plot_pca_scores(pca_results, df, output_dir, pc_x=1, pc_y=2, file_name="pca_score_plot", class_col='Class',
                    show_ellipse=True, is_sum_pca=False):
    """
    Generates PCA Score Plot with optional 95% Confidence Ellipses (Hotelling's T2).
    """
    os.makedirs(output_dir, exist_ok=True)

    scores = pca_results['scores']
    var_ratio = pca_results['explained_variance']
    pc_x_label, pc_y_label = f"PC{pc_x}", f"PC{pc_y}"

    x_data = scores[pc_x_label]
    y_data = scores[pc_y_label]

    fig, ax = plt.subplots()

    mask_visible = df[class_col] != 'NotShow'
    visible_classes = df.loc[mask_visible, class_col].unique()
    current_palette = DISCRETE_COLORS[:len(visible_classes)]
    markers_dict = {cls: MARKERS[i % len(MARKERS)] for i, cls in enumerate(visible_classes)}

    # Scatter Plot
    sns.scatterplot(x=x_data[mask_visible], y=y_data[mask_visible],
                    hue=df.loc[mask_visible, class_col], style=df.loc[mask_visible, class_col],
                    palette=current_palette, markers=markers_dict,
                    s=100, alpha=0.8, edgecolor='k', ax=ax)

    # Confidence Ellipses (per class)
    if show_ellipse:
        for i, cls in enumerate(visible_classes):
            cls_mask = (df[class_col] == cls) & mask_visible
            x_cls, y_cls = x_data[cls_mask], y_data[cls_mask]

            if len(x_cls) > 2:
                cov = np.cov(x_cls, y_cls)
                mean_x, mean_y = np.mean(x_cls), np.mean(y_cls)
                lambda_, v = np.linalg.eig(cov)

                # Sort eigenvalues
                order = lambda_.argsort()[::-1]
                lambda_, v = lambda_[order], v[:, order]
                angle = np.degrees(np.arctan2(*v[:, 0][::-1]))

                # 95% Confidence (Chi-square, 2 dof = 5.991)
                scale_factor = 2 * np.sqrt(5.991)
                width, height = scale_factor * np.sqrt(lambda_)

                ellipse = Ellipse(xy=(mean_x, mean_y), width=width, height=height, angle=angle,
                                  facecolor='none', edgecolor=current_palette[i], linestyle='--', linewidth=1.5)
                ax.add_patch(ellipse)

    var_x, var_y = var_ratio[pc_x - 1] * 100, var_ratio[pc_y - 1] * 100
    ax.set_xlabel(f"{pc_x_label} ({var_x:.1f}%)", fontweight='bold')
    ax.set_ylabel(f"{pc_y_label} ({var_y:.1f}%)", fontweight='bold')

    title_text = f"SUM-PCA Super Scores: {pc_x_label} vs {pc_y_label}" if is_sum_pca else f"PCA Score Plot: {pc_x_label} vs {pc_y_label}"
    ax.set_title(title_text, fontweight='bold')
    ax.axhline(0, color='grey', linestyle=':', linewidth=0.8)
    ax.axvline(0, color='grey', linestyle=':', linewidth=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_name}_PC{pc_x}vsPC{pc_y}.{SAVE_FORMAT}"), format=SAVE_FORMAT,
                bbox_inches='tight')
    plt.close(fig)


def plot_pca_loadings(pca_results, output_dir, pc_x=1, pc_y=2, file_name="pca_loading_plot", top_n=20):
    """
    Generates PCA Loading Plot to identify variables driving the separation.
    """
    os.makedirs(output_dir, exist_ok=True)

    loadings = pca_results['loadings']
    pc_x_label, pc_y_label = f"PC{pc_x}", f"PC{pc_y}"

    # Calculate magnitude
    loadings['magnitude'] = np.sqrt(loadings[pc_x_label] ** 2 + loadings[pc_y_label] ** 2)
    top_loadings = loadings.sort_values(by='magnitude', ascending=False).head(top_n)

    fig, ax = plt.subplots()

    # All features (background)
    ax.scatter(loadings[pc_x_label], loadings[pc_y_label], color=DISCRETE_COLORS[3], alpha=0.7, s=20,
               label='All Features')

    # Top features (highlighted)
    if top_n > 0:
        ax.scatter(top_loadings[pc_x_label], top_loadings[pc_y_label], color=DISCRETE_COLORS[9], s=50,
                   label=f'Top {top_n} Contributors')

    # Annotations
    for feature, row in top_loadings.iterrows():
        plt.text(row[pc_x_label], row[pc_y_label], feature, fontsize=8, color=DISCRETE_COLORS[9], fontweight='bold')

    ax.axhline(0, color='k', linestyle='-', linewidth=0.8)
    ax.axvline(0, color='k', linestyle='-', linewidth=0.8)

    max_val = max(abs(loadings[pc_x_label].max()), abs(loadings[pc_y_label].max())) * 1.1
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)

    ax.set_xlabel(f"Loading {pc_x_label}", fontweight='bold')
    ax.set_ylabel(f"Loading {pc_y_label}", fontweight='bold')
    ax.set_title(f"PCA Loading Plot: {pc_x_label} vs {pc_y_label}", fontweight='bold')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_name}_PC{pc_x}vsPC{pc_y}.{SAVE_FORMAT}"), format=SAVE_FORMAT,
                bbox_inches='tight')
    plt.close(fig)


def plot_pca_scree(pca_results, output_dir, file_name="pca_scree_plot", threshold=0.9):
    """
    Generates Scree Plot (Explained Variance vs PC Index) to determine optimal components.
    """
    os.makedirs(output_dir, exist_ok=True)

    var_ratio = pca_results['explained_variance']
    cum_var = pca_results['cumulative_variance']
    n_plot = min(20, len(var_ratio))
    x_range = np.arange(1, n_plot + 1)

    fig, ax1 = plt.subplots()

    # Bars (Individual)
    ax1.bar(x_range, var_ratio[:n_plot] * 100, color=DISCRETE_COLORS[0], alpha=0.7, label='Individual Variance')
    ax1.set_xlabel('Principal Components')
    ax1.set_ylabel('Explained Variance (%)', color=DISCRETE_COLORS[0])
    ax1.tick_params(axis='y', labelcolor=DISCRETE_COLORS[0])
    ax1.set_xticks(x_range)

    # Line (Cumulative)
    ax2 = ax1.twinx()
    ax2.plot(x_range, cum_var[:n_plot] * 100, color=DISCRETE_COLORS[4], marker='o', linewidth=2,
             label='Cumulative Variance')
    ax2.set_ylabel('Cumulative Variance (%)', color=DISCRETE_COLORS[4])
    ax2.tick_params(axis='y', labelcolor=DISCRETE_COLORS[4])

    # Threshold
    ax2.axhline(y=threshold * 100, color='grey', linestyle='--', alpha=0.5)
    ax2.text(n_plot, threshold * 100 + 1, f'{int(threshold * 100)}% Threshold', color='grey', ha='right')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')

    plt.title('PCA Scree Plot', fontweight='bold')
    plt.savefig(os.path.join(output_dir, f"{file_name}.{SAVE_FORMAT}"), format=SAVE_FORMAT, bbox_inches='tight')
    plt.close(fig)


def plot_sum_pca_loadings(pca_results, output_dir, pc_x=1, pc_y=2, file_name="sum_pca_loadings_blocks", top_n=20):
    """
    Generates SUM-PCA Loading Plot, coloring features by their Source Block.
    """
    os.makedirs(output_dir, exist_ok=True)

    loadings = pca_results['loadings'].copy()
    pc_x_label, pc_y_label = f"PC{pc_x}", f"PC{pc_y}"

    # Block Identification Logic
    def identify_block(feature_name):
        if "_Block1" in feature_name:
            return "Block 1 (Neg)"
        elif "_Block2" in feature_name:
            return "Block 2 (Pos)"
        return "Unknown"

    loadings['Block'] = loadings.index.to_series().apply(identify_block)
    loadings['magnitude'] = np.sqrt(loadings[pc_x_label] ** 2 + loadings[pc_y_label] ** 2)

    top_loadings = loadings.sort_values(by='magnitude', ascending=False).head(top_n)
    unique_blocks = sorted(loadings['Block'].unique())
    block_palette = DISCRETE_COLORS[:len(unique_blocks)]

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.scatterplot(data=loadings, x=pc_x_label, y=pc_y_label, hue='Block', style='Block',
                    palette=block_palette, alpha=0.7, s=40, edgecolor='none', ax=ax)

    # Annotation
    color_dict = dict(zip(unique_blocks, block_palette))
    for feature, row in top_loadings.iterrows():
        clean_name = feature.split('_Block')[0]
        ax.text(row[pc_x_label], row[pc_y_label], clean_name, fontsize=9,
                color=color_dict.get(row['Block'], 'black'), fontweight='bold')

    max_val = max(abs(loadings[pc_x_label].max()), abs(loadings[pc_y_label].max())) * 1.1
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.axhline(0, color='k', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.axvline(0, color='k', linestyle='-', linewidth=0.8, alpha=0.5)

    var_x = pca_results['explained_variance'][pc_x - 1] * 100
    var_y = pca_results['explained_variance'][pc_y - 1] * 100
    ax.set_xlabel(f"Super Loading {pc_x_label} ({var_x:.1f}%)", fontweight='bold')
    ax.set_ylabel(f"Super Loading {pc_y_label} ({var_y:.1f}%)", fontweight='bold')
    ax.set_title(f"SUM-PCA Block Contribution: {pc_x_label} vs {pc_y_label}", fontweight='bold')
    ax.legend(title="Data Block")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_name}_PC{pc_x}vsPC{pc_y}.{SAVE_FORMAT}"), format=SAVE_FORMAT,
                bbox_inches='tight')
    plt.close(fig)


def plot_sum_pca_scree_contribution(pca_results, output_dir, file_name="sum_pca_scree_blocks_grouped", threshold=0.9):
    """
    Generates Grouped Scree Plot for SUM-PCA showing contribution of each block per PC.
    """
    os.makedirs(output_dir, exist_ok=True)

    var_ratio = pca_results['explained_variance']
    cum_var = pca_results['cumulative_variance']
    loadings_df = pca_results['loadings']
    n_plot = min(20, len(var_ratio))
    x = np.arange(n_plot)
    width = 0.35

    var_b1, var_b2 = [], []

    for i in range(n_plot):
        pc_name = f"PC{i + 1}"
        pc_loadings = loadings_df[pc_name]

        sq_load_b1 = pc_loadings[pc_loadings.index.str.contains("_Block1")].pow(2).sum()
        sq_load_b2 = pc_loadings[pc_loadings.index.str.contains("_Block2")].pow(2).sum()
        total_sq = sq_load_b1 + sq_load_b2

        var_b1.append((sq_load_b1 / total_sq) * var_ratio[i] * 100)
        var_b2.append((sq_load_b2 / total_sq) * var_ratio[i] * 100)

    fig, ax1 = plt.subplots(figsize=(12, 7))

    ax1.bar(x - width / 2, var_b1, width, label='Block 1 (Neg)', color=DISCRETE_COLORS[0], alpha=0.8)
    ax1.bar(x + width / 2, var_b2, width, label='Block 2 (Pos)', color=DISCRETE_COLORS[1], alpha=0.8)

    ax1.set_xlabel('Principal Components (Super Scores)')
    ax1.set_ylabel('Explained Variance (%)')
    ax1.set_title('SUM-PCA: Variance Contribution per Block', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"PC{i + 1}" for i in range(n_plot)])

    ax2 = ax1.twinx()
    ax2.plot(x, cum_var[:n_plot] * 100, color='black', marker='o', linewidth=2, label='Cumulative Variance')
    ax2.set_ylabel('Cumulative Variance (%)')
    ax2.set_ylim(0, 105)
    ax2.axhline(y=threshold * 100, color='grey', linestyle='--', alpha=0.5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_name}.{SAVE_FORMAT}"), format=SAVE_FORMAT, bbox_inches='tight')
    plt.close(fig)


def detect_pca_outliers(pca_results, conf_level=0.95):
    """
    Performs PCA-based Anomaly Detection (Hotelling's T2 + Q-Residuals).
    """
    scores_all = pca_results['scores'].values
    loadings_all = pca_results['loadings'].values
    eigenvalues_all = pca_results['model'].explained_variance_
    X_scaled = pca_results['data_scaled']

    n_samples = scores_all.shape[0]
    total_components = scores_all.shape[1]

    # Dynamic Component Selection (90% variance)
    explained_var_ratio = eigenvalues_all / np.sum(eigenvalues_all)
    cum_var = np.cumsum(explained_var_ratio)

    target_var = 0.90
    n_components_var = np.argmax(cum_var >= target_var) + 1
    max_safe_components = max(1, n_samples - 2)  # Prevent Division by Zero
    n_comp_final = min(n_components_var, max_safe_components, total_components)
    if n_comp_final < 1: n_comp_final = 1

    scores = scores_all[:, :n_comp_final]
    loadings = loadings_all[:, :n_comp_final]
    eigenvalues = eigenvalues_all[:n_comp_final]

    # T2 (Hotelling)
    t2_values = np.sum((scores ** 2) / eigenvalues, axis=1)

    d1, d2 = n_comp_final, n_samples - n_comp_final
    if d2 <= 0:
        t2_limit = np.inf
    else:
        F_crit = f.ppf(conf_level, d1, d2)
        t2_limit = (d1 * (n_samples - 1) / d2) * F_crit

    # Q-Residuals
    X_reconstructed = np.dot(scores, loadings.T)
    E_matrix = X_scaled - X_reconstructed
    q_values = np.sum(E_matrix ** 2, axis=1)

    mean_q, var_q = np.mean(q_values), np.var(q_values)
    if var_q > 0:
        g = var_q / (2 * mean_q)
        h = (2 * mean_q ** 2) / var_q
        q_limit = g * chi2.ppf(conf_level, df=h)
    else:
        q_limit = 0.0

    results_df = pd.DataFrame(index=pca_results['scores'].index)
    results_df['T2'] = t2_values
    results_df['T2_Limit'] = t2_limit
    results_df['Q_Residuals'] = q_values
    results_df['Q_Limit'] = q_limit
    results_df['Outlier_T2'] = results_df['T2'] > t2_limit
    results_df['Outlier_Q'] = results_df['Q_Residuals'] > q_limit
    results_df['is_outlier'] = results_df['Outlier_T2'] | results_df['Outlier_Q']

    def classify(row):
        if row['Outlier_T2'] and row['Outlier_Q']: return "Both"
        if row['Outlier_T2']: return "Hotelling (Extreme)"
        if row['Outlier_Q']: return "Q-Res (Model Mismatch)"
        return "Normal"

    results_df['Outlier_Type'] = results_df.apply(classify, axis=1)
    return results_df


def plot_distance_plot(pca_results, df, output_dir, highlight_samples=None, file_name="Distance_Plot"):
    """
    Generates Distance Plot (T2 vs Q) for outlier diagnosis.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Re-calculate metrics for plot consistency
    scores = pca_results['scores'].values
    loadings = pca_results['loadings'].values
    eigenvalues = pca_results['model'].explained_variance_
    X_scaled = pca_results['data_scaled']

    cum_var = np.cumsum(eigenvalues / np.sum(eigenvalues))
    n_comp = np.argmax(cum_var >= 0.90) + 1

    scores_sub = scores[:, :n_comp]
    loadings_sub = loadings[:, :n_comp]
    eigenvals_sub = eigenvalues[:n_comp]

    T2 = np.sum((scores_sub ** 2) / eigenvals_sub, axis=1)
    X_reconstructed = np.dot(scores_sub, loadings_sub.T)
    Q = np.sum((X_scaled - X_reconstructed) ** 2, axis=1)

    n = df.shape[0]
    conf = 0.95
    F_crit = f.ppf(conf, n_comp, n - n_comp)
    T2_lim = (n_comp * (n - 1) / (n - n_comp)) * F_crit

    mean_q, var_q = np.mean(Q), np.var(Q)
    g = var_q / (2 * mean_q)
    h = (2 * mean_q ** 2) / var_q
    Q_lim = g * chi2.ppf(conf, df=h)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(T2, Q, c='grey', alpha=0.5, label='Samples')

    if highlight_samples:
        indices_to_plot = [i for i, name in enumerate(df.index) if name in highlight_samples]
        if indices_to_plot:
            ax.scatter(T2[indices_to_plot], Q[indices_to_plot], c='red', s=100, label='Out Candidates')
            for idx in indices_to_plot:
                ax.text(T2[idx], Q[idx], df.index[idx], fontsize=9, color='red', fontweight='bold')

    ax.axvline(T2_lim, color='blue', linestyle='--', label=f'T2 Limit (95%)')
    ax.axhline(Q_lim, color='green', linestyle='--', label=f'Q Limit (95%)')

    ax.set_xlabel("Hotelling's T2", fontweight='bold')
    ax.set_ylabel("Q-Residuals", fontweight='bold')
    ax.set_title(f"Distance Plot: {file_name}", fontweight='bold')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_name}.pdf"))
    plt.close()