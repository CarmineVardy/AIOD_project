"""
PLS-DA (Partial Least Squares Discriminant Analysis) Module.

This module provides wrappers for training PLS-DA models, optimizing latent variables (LVs),
extracting Variable Importance in Projection (VIP) scores, and performing Permutation Tests
to validate model significance.

It includes dedicated visualization functions for Score Plots, Loading Plots,
and Predicted vs Observed plots.
"""

import os

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedKFold, permutation_test_score
from sklearn.preprocessing import LabelEncoder

from src.config_visualization import *


def run_pls_da(X_train, y_train, X_test, n_components=2, threshold=0.5):
    """
    Trains a PLS-DA model.
    PLS-DA uses regression logic (dummy-encoded Y) for classification.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_test (pd.DataFrame): Test features.
        n_components (int): Number of Latent Variables (LVs).
        threshold (float): Decision threshold (default 0.5).

    Returns:
        tuple: (y_pred, y_prob, feature_info, model_results)
    """
    # 1. Encode Labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)

    # 2. Fit PLS Regression (Assuming external scaling)
    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(X_train, y_train_enc)

    # 3. Prediction (Continuous output)
    y_pred_continuous = pls.predict(X_test).flatten()

    # Align probabilities to Positive Class
    if le.classes_[0] == 'CHD':
        y_prob_for_roc = 1 - y_pred_continuous
    else:
        y_prob_for_roc = y_pred_continuous

    # 4. Classification (Thresholding)
    y_pred_enc = (y_pred_continuous >= threshold).astype(int)
    y_pred = le.inverse_transform(y_pred_enc)

    # 5. Feature Selection (VIPs & Coefficients)
    vip_scores = _calculate_vips(pls)
    coefficients = pls.coef_.flatten()

    feature_names = X_train.columns
    vip_dict = dict(zip(feature_names, vip_scores))

    feature_info = {
        'coefficients': coefficients,
        'vip_scores': vip_scores,
        'feature_importance_ranking': dict(
            sorted(vip_dict.items(), key=lambda item: item[1], reverse=True)
        )
    }

    # 6. Store Data for Plotting
    model_results = {
        'model': pls,
        'train_x_scores': pd.DataFrame(
            pls.x_scores_, index=X_train.index,
            columns=[f'LV{i + 1}' for i in range(n_components)]
        ),
        'train_y_enc': y_train_enc,
        'class_names': le.classes_,
        'feature_names': feature_names
    }

    return y_pred, y_prob_for_roc, feature_info, model_results


def _calculate_vips(model):
    """
    Calculates VIP (Variable Importance in Projection) scores.
    VIP represents the importance of each feature in projecting X onto the latent space
    while maximizing covariance with Y.
    """
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_

    p, h = w.shape
    vips = np.zeros((p,))

    # Explained variance of Y per component
    s = np.diag(t.T @ t @ q.T @ q).reshape(h)
    total_s = np.sum(s)

    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)])
        vips[i] = np.sqrt(p * (s.T @ weight) / total_s)

    return vips


def run_permutation_test(X, y, n_components=2, n_permutations=100):
    """
    Performs Permutation Test.
    Shuffles labels N times to estimate the distribution of random performance
    and calculates a p-value for the real model's score.
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    def pls_accuracy_scorer(estimator, X, y_true):
        y_pred = (estimator.predict(X).flatten() >= 0.5).astype(int)
        return np.mean(y_pred == y_true)

    pls = PLSRegression(n_components=n_components, scale=False)

    print(f"Running Permutation Test ({n_permutations} iterations)...")
    score, perm_scores, pvalue = permutation_test_score(
        pls, X, y_enc, scoring=pls_accuracy_scorer, cv=StratifiedKFold(5),
        n_permutations=n_permutations, n_jobs=-1
    )

    return score, perm_scores, pvalue


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_pls_scores(model_results, output_dir, lv_x=1, lv_y=2, file_name="pls_score_plot"):
    """
    Generates PLS-DA Score Plot showing sample clustering in latent space.
    """
    os.makedirs(output_dir, exist_ok=True)

    scores = model_results['train_x_scores']
    y_enc = model_results['train_y_enc']
    class_names = model_results['class_names']
    labels = [class_names[val] for val in y_enc]

    lv_x_label = f"LV{lv_x}"
    lv_y_label = f"LV{lv_y}"

    x_data = scores[lv_x_label]
    y_data = scores[lv_y_label]

    unique_classes = np.unique(labels)
    current_palette = DISCRETE_COLORS[:len(unique_classes)]
    markers_dict = {cls: MARKERS[i % len(MARKERS)] for i, cls in enumerate(unique_classes)}

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.scatterplot(x=x_data, y=y_data, hue=labels, style=labels,
                    palette=current_palette, markers=markers_dict,
                    s=100, alpha=0.9, edgecolor='k', ax=ax)

    ax.set_xlabel(f"{lv_x_label}", fontweight='bold')
    ax.set_ylabel(f"{lv_y_label}", fontweight='bold')
    ax.set_title(f"PLS-DA Score Plot: {lv_x_label} vs {lv_y_label}", fontweight='bold')
    ax.axhline(0, color='grey', linestyle=':', linewidth=0.8)
    ax.axvline(0, color='grey', linestyle=':', linewidth=0.8)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{file_name}_LV{lv_x}vsLV{lv_y}.{SAVE_FORMAT}")
    plt.savefig(save_path, format=SAVE_FORMAT, bbox_inches='tight')
    plt.close(fig)


def plot_pls_classification_error(optimization_df, output_dir, file_name="pls_optimization"):
    """
    Plots Classification Error vs Number of LVs to identify the optimal complexity.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    x = optimization_df['n_components']
    y_err = optimization_df['Error']

    ax.plot(x, y_err, marker='o', color=DISCRETE_COLORS[0], linewidth=2, label='CV Error')

    best_n = optimization_df.loc[optimization_df['Error'].idxmin(), 'n_components']
    ax.axvline(best_n, color='red', linestyle='--', alpha=0.6, label=f'Optimal LVs ({int(best_n)})')

    ax.set_title("PLS-DA Component Optimization", fontweight='bold')
    ax.set_xlabel("Number of Latent Variables (LVs)")
    ax.set_ylabel("Classification Error")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{file_name}.{SAVE_FORMAT}")
    plt.savefig(save_path, format=SAVE_FORMAT, bbox_inches='tight')
    plt.close(fig)


def plot_pls_predicted_vs_observed(y_true, y_pred_continuous, output_dir, file_name="pls_pred_vs_obs"):
    """
    Scatter plot of continuous predictions vs observed classes.
    Visualizes the decision boundary and misclassified samples.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    indices = np.arange(len(y_true))
    unique_classes = np.unique(y_true)
    current_palette = DISCRETE_COLORS[:len(unique_classes)]

    sns.scatterplot(x=indices, y=y_pred_continuous, hue=y_true, palette=current_palette,
                    s=80, edgecolor='k', ax=ax)

    ax.axhline(0.5, color='red', linestyle='--', label='Threshold (0.5)')

    # Highlight classification zones
    ax.axhspan(0.5, max(y_pred_continuous.max(), 1.1), color=current_palette[1], alpha=0.1)
    ax.axhspan(min(y_pred_continuous.min(), -0.1), 0.5, color=current_palette[0], alpha=0.1)

    ax.set_title("PLS-DA: Predicted vs Observed", fontweight='bold')
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Predicted Score")
    ax.legend(loc='lower right')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{file_name}.{SAVE_FORMAT}")
    plt.savefig(save_path, format=SAVE_FORMAT, bbox_inches='tight')
    plt.close(fig)


def plot_pls_loadings(model_results, output_dir, lv_x=1, lv_y=2, file_name="pls_loadings_plot", top_n=20):
    """
    Generates PLS Loadings Plot.
    Highlights top contributing features based on vector magnitude.
    """
    os.makedirs(output_dir, exist_ok=True)

    model = model_results['model']
    feature_names = model_results['feature_names']
    loadings_mat = model.x_loadings_

    cols = [f"LV{i + 1}" for i in range(loadings_mat.shape[1])]
    loadings_df = pd.DataFrame(loadings_mat, index=feature_names, columns=cols)

    lv_x_label = f"LV{lv_x}"
    lv_y_label = f"LV{lv_y}"

    magnitude = np.sqrt(loadings_df[lv_x_label] ** 2 + loadings_df[lv_y_label] ** 2)
    loadings_df['magnitude'] = magnitude
    top_loadings = loadings_df.sort_values(by='magnitude', ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(loadings_df[lv_x_label], loadings_df[lv_y_label], color=DISCRETE_COLORS[3], alpha=0.4, s=20,
               label='All Features')
    ax.scatter(top_loadings[lv_x_label], top_loadings[lv_y_label], color=DISCRETE_COLORS[0], s=50,
               label=f'Top {top_n} Contributors')

    for feature, row in top_loadings.iterrows():
        plt.text(row[lv_x_label], row[lv_y_label], feature, fontsize=8, color=DISCRETE_COLORS[0], fontweight='bold')

    ax.axhline(0, color='k', linestyle='-', linewidth=0.8)
    ax.axvline(0, color='k', linestyle='-', linewidth=0.8)

    max_val = max(abs(loadings_df[lv_x_label].max()), abs(loadings_df[lv_y_label].max())) * 1.1
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)

    ax.set_xlabel(f"Loading {lv_x_label}", fontweight='bold')
    ax.set_ylabel(f"Loading {lv_y_label}", fontweight='bold')
    ax.set_title(f"PLS-DA Loadings: {lv_x_label} vs {lv_y_label}", fontweight='bold')
    ax.legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{file_name}_LV{lv_x}vsLV{lv_y}.{SAVE_FORMAT}")
    plt.savefig(save_path, format=SAVE_FORMAT, bbox_inches='tight')
    plt.close(fig)