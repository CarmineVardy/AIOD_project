"""
Anomaly Detection Module.

This module provides implementations for unsupervised anomaly detection algorithms:
1. One-Class SVM (Geometric approach)
2. Isolation Forest (Random partitioning approach)
3. Local Outlier Factor (Density-based approach)

It also includes a visualization function to project decision boundaries onto a 2D PCA space
for interpretability.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


def run_one_class_svm(X, nu=0.1, kernel="rbf", gamma=0.1):
    """
    Runs One-Class SVM.
    Methodology: Maps data to high-dimensional space and finds the smallest hypersphere
    enclosing the 'normal' observations.

    Args:
        X (pd.DataFrame or np.array): Input features.
        nu (float): Upper bound on the fraction of training errors/outliers.
        kernel (str): Kernel type (default 'rbf' for non-linear boundaries).
        gamma (float): Kernel coefficient.

    Returns:
        dict: Contains the trained model, predictions (-1=outlier, 1=inlier), and scores.
    """
    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    y_pred = model.fit_predict(X)

    return {
        'model': model,
        'y_pred': y_pred,
        'scores': model.decision_function(X)  # Distance to the separating hyperplane
    }


def run_isolation_forest(X, contamination=0.1, random_state=42):
    """
    Runs Isolation Forest.
    Methodology: Builds an ensemble of random trees. Anomalies are isolated in fewer steps
    (shorter path length) compared to normal points. Robust to high dimensionality.

    Args:
        X (pd.DataFrame or np.array): Input features.
        contamination (float): Expected proportion of outliers in the dataset.
        random_state (int): Seed for reproducibility.

    Returns:
        dict: Contains the trained model, predictions, and anomaly scores.
    """
    model = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
    y_pred = model.fit_predict(X)

    return {
        'model': model,
        'y_pred': y_pred,
        'scores': model.decision_function(X)  # Mean anomaly score (lower is more abnormal)
    }


def run_local_outlier_factor(X, n_neighbors=20, contamination=0.1):
    """
    Runs Local Outlier Factor (LOF).
    Methodology: Compares the local density of a sample to the density of its k-nearest neighbors.
    Points with significantly lower density than their neighbors are flagged as outliers.

    Args:
        X (pd.DataFrame or np.array): Input features.
        n_neighbors (int): Number of neighbors to use for density estimation.
        contamination (float): Expected proportion of outliers.

    Returns:
        dict: Contains the trained model, predictions, and negative LOF scores.
    """
    # novelty=True enables use on new data (predict) and decision_function, treating X as training set.
    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=True)
    model.fit(X)
    y_pred = model.predict(X)

    return {
        'model': model,
        'y_pred': y_pred,
        'scores': model.decision_function(X)
    }


def plot_anomaly_comparison(X_original, filename, output_dir):
    """
    Generates a 2D comparative visualization of decision boundaries for the three algorithms.
    Projects the high-dimensional input onto the first two Principal Components.

    Args:
        X_original (pd.DataFrame): The dataset used for anomaly detection.
        filename (str): Name tag for the output file (e.g., 'CTRL_Group').
        output_dir (str): Directory to save the plot.
    """
    # 1. Project to 2D for Visualization
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_original)

    # Variance explained for axis labels
    var_pc1 = pca.explained_variance_ratio_[0] * 100
    var_pc2 = pca.explained_variance_ratio_[1] * 100

    # 2. Define Algorithms Configuration (Re-instantiated for 2D fitting)
    algorithms = [
        ("One-Class SVM", OneClassSVM(nu=0.05, kernel="rbf", gamma=0.1)),
        ("Isolation Forest", IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)),
        ("Local Outlier Factor", LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True))
    ]

    # 3. Setup Grid
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    plt.subplots_adjust(left=.05, right=.95, wspace=0.3, bottom=0.25)
    axs = axs.flatten()

    # 4. Create Meshgrid for Contour Plots (Decision Surface)
    x_range = X_2d[:, 0].max() - X_2d[:, 0].min()
    y_range = X_2d[:, 1].max() - X_2d[:, 1].min()
    x_pad, y_pad = x_range * 0.2, y_range * 0.2

    x_min, x_max = X_2d[:, 0].min() - x_pad, X_2d[:, 0].max() + x_pad
    y_min, y_max = X_2d[:, 1].min() - y_pad, X_2d[:, 1].max() + y_pad

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150),
                         np.linspace(y_min, y_max, 150))

    # 5. Iterative Plotting per Algorithm
    for i, (name, algorithm) in enumerate(algorithms):
        ax = axs[i]

        try:
            algorithm.fit(X_2d)
            y_pred = algorithm.predict(X_2d)

            # Calculate Decision Function for Contours
            if hasattr(algorithm, "decision_function"):
                Z = algorithm.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = np.zeros(xx.ravel().shape)
            Z = Z.reshape(xx.shape)

            # Draw Boundaries (Solid line for 0 level, dashed for confidence regions)
            if Z.min() < 0 < Z.max():
                ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black', linestyles='-')

            end_level = min(0, Z.max())
            if Z.min() < end_level:
                levels = np.linspace(Z.min(), end_level, 5)
                ax.contour(xx, yy, Z, levels=levels, linewidths=0.8, colors='black', alpha=0.3, linestyles='--')

            # Scatter Plot: Normal Data (Purple Circles) vs Outliers (Gold Crosses)
            ax.scatter(X_2d[y_pred == 1, 0], X_2d[y_pred == 1, 1],
                       s=30, c='#4B0082', marker='o', edgecolors='k', alpha=0.7,
                       label='Normal Data' if i == 0 else "")

            ax.scatter(X_2d[y_pred == -1, 0], X_2d[y_pred == -1, 1],
                       s=60, c='gold', marker='X', edgecolors='k', linewidth=1.2,
                       label='Outlier' if i == 0 else "")

            # Labels and Aesthetics
            ax.set_title(name, fontsize=14, fontweight='bold', pad=10)
            ax.set_xlabel(f"PC1 ({var_pc1:.1f}%)", fontsize=11, fontweight='bold')
            ax.set_ylabel(f"PC2 ({var_pc2:.1f}%)", fontsize=11, fontweight='bold')
            ax.grid(True, linestyle=':', alpha=0.4)

        except Exception as e:
            ax.text(0.5, 0.5, f"Error:\n{str(e)}", ha='center')

    # Global Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Normal Data',
               markerfacecolor='#4B0082', markersize=10, markeredgecolor='k'),
        Line2D([0], [0], marker='X', color='w', label='Detected Outlier',
               markerfacecolor='gold', markersize=12, markeredgecolor='k'),
        Line2D([0], [0], color='black', lw=2, label='Decision Boundary (Limit)')
    ]

    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=12,
               frameon=True, edgecolor='black', bbox_to_anchor=(0.5, 0.05))

    fig.suptitle(f"Algorithmic Decision Boundaries (Projected on PCA Space): {filename}",
                 fontsize=16, fontweight='bold', y=0.98)

    # Save
    final_name = f"Comparison_Grid_{filename}.pdf"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, final_name)
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close(fig)