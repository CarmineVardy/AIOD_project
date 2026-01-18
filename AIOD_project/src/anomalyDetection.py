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
    Theory: Finds the smallest hypersphere enclosing the 'normal' data.
    Robust to non-linear distributions via Kernel Trick.
    """
    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)

    # Fit and Predict (-1 = outlier, 1 = inlier)
    y_pred = model.fit_predict(X)

    return {
        'model': model,
        'y_pred': y_pred,
        'scores': model.decision_function(X)  # Distance to the separating hyperplane
    }


def run_isolation_forest(X, contamination=0.1, random_state=42):
    """
    Runs Isolation Forest.
    Theory: Anomalies are 'few and different', thus easy to isolate (short path length).
    Handles 'clustered outliers' better than density methods.
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
    Theory: Compares local density of a point vs its neighbors (k-NN).

    Note: We use novelty=True to enable 'decision_function' for plotting contours,
    treating the training set as the reference for normality.
    """
    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=True)

    model.fit(X)

    # With novelty=True, we use predict() on X
    y_pred = model.predict(X)

    return {
        'model': model,
        'y_pred': y_pred,
        'scores': model.decision_function(X)  # Negative LOF score
    }


# =============================================================================
# VISUALIZATION FUNCTION
# =============================================================================

def plot_anomaly_comparison(X_original, filename, output_dir):
    """
    Generates a comparison grid for the 3 selected algorithms according to strict
    thesis formatting guidelines (Labels on all axes, Clear Titles, B&W compatible symbols).
    """

    # 1. Project to 2D for Visualization
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_original)

    # Calculate Variance for Axis Labels
    var_pc1 = pca.explained_variance_ratio_[0] * 100
    var_pc2 = pca.explained_variance_ratio_[1] * 100

    # 2. Define Algorithms Configuration
    # We instantiate new models here to fit on the 2D data for visualization purposes
    algorithms = [
        ("One-Class SVM", OneClassSVM(nu=0.05, kernel="rbf", gamma=0.1)),  # Nu 0.05 coerente con main
        ("Isolation Forest", IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)),
        ("Local Outlier Factor", LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True))
    ]

    # 3. Setup Grid (1 Row, 3 Columns)
    # Aumentiamo la dimensione verticale per dare spazio alle etichette
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    # Adjust spacing: wspace=0.3 garantisce spazio laterale tra i grafici per le label Y
    # bottom=0.25 garantisce spazio sotto per la legenda globale
    plt.subplots_adjust(left=.05, right=.95, wspace=0.3, bottom=0.25)
    axs = axs.flatten()

    # 4. Dynamic Padding Logic (Meshgrid creation)
    x_range = X_2d[:, 0].max() - X_2d[:, 0].min()
    y_range = X_2d[:, 1].max() - X_2d[:, 1].min()
    x_pad, y_pad = x_range * 0.2, y_range * 0.2
    x_min, x_max = X_2d[:, 0].min() - x_pad, X_2d[:, 0].max() + x_pad
    y_min, y_max = X_2d[:, 1].min() - y_pad, X_2d[:, 1].max() + y_pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150),
                         np.linspace(y_min, y_max, 150))

    # 5. Iterative Plotting
    for i, (name, algorithm) in enumerate(algorithms):
        ax = axs[i]

        try:
            algorithm.fit(X_2d)

            # --- Prediction ---
            if name == "Local Outlier Factor":
                y_pred = algorithm.predict(X_2d)
            else:
                y_pred = algorithm.predict(X_2d)

            # --- Z-Surface (Decision Function) ---
            if hasattr(algorithm, "decision_function"):
                Z = algorithm.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = np.zeros(xx.ravel().shape)
            Z = Z.reshape(xx.shape)

            # --- Drawing Contours ---
            if Z.min() < 0 < Z.max():
                ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black', linestyles='-')
            end_level = min(0, Z.max())
            if Z.min() < end_level:
                levels = np.linspace(Z.min(), end_level, 5)
                ax.contour(xx, yy, Z, levels=levels, linewidths=0.8, colors='black', alpha=0.3, linestyles='--')

            # --- Scatter Plot (B&W Friendly Symbols) ---
            # Inliers: Circles (o), Outliers: Crosses (X)
            # Colors help, but symbols ensure B&W readability.

            # Inliers
            ax.scatter(X_2d[y_pred == 1, 0], X_2d[y_pred == 1, 1],
                       s=30, c='#4B0082', marker='o', edgecolors='k', alpha=0.7,
                       label='Normal Data' if i == 0 else "")  # Label only once for internal legend trick

            # Outliers
            ax.scatter(X_2d[y_pred == -1, 0], X_2d[y_pred == -1, 1],
                       s=60, c='gold', marker='X', edgecolors='k', linewidth=1.2,
                       label='Outlier' if i == 0 else "")

            # --- COMPLIANCE FIX: Labels on EVERY graph ---
            ax.set_title(name, fontsize=14, fontweight='bold', pad=10)

            # X Axis Label (Always present)
            ax.set_xlabel(f"PC1 ({var_pc1:.1f}%)", fontsize=11, fontweight='bold')

            # Y Axis Label (Always present now)
            ax.set_ylabel(f"PC2 ({var_pc2:.1f}%)", fontsize=11, fontweight='bold')

            # Add Grid (Good practice for readability)
            ax.grid(True, linestyle=':', alpha=0.4)

        except Exception as e:
            ax.text(0.5, 0.5, f"Error:\n{str(e)}", ha='center')
            print(f"Error plotting {name}: {e}")

    # Global Legend (Moved further down to be distinct)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Normal Data',
               markerfacecolor='#4B0082', markersize=10, markeredgecolor='k'),
        Line2D([0], [0], marker='X', color='w', label='Detected Outlier',
               markerfacecolor='gold', markersize=12, markeredgecolor='k'),
        Line2D([0], [0], color='black', lw=2, label='Decision Boundary (Limit)')
    ]

    # Legend posizionata bene sotto, centrata
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=12,
               frameon=True, edgecolor='black', bbox_to_anchor=(0.5, 0.05))

    fig.suptitle(f"Algorithmic Decision Boundaries (Projected on PCA Space): {filename}",
                 fontsize=16, fontweight='bold', y=0.98)

    # --- SAVE LOGIC INLINE ---
    final_name = f"Comparison_Grid_{filename}.pdf"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, final_name)
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Graph {final_name} saved in: {output_dir}")