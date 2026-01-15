import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.covariance import EllipticEnvelope
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
from sklearn.linear_model import SGDOneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.kernel_approximation import Nystroem
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

class AnomalyDetector:
    """
    A class dedicated to detecting anomalies in metabolomic datasets using various
    statistical and machine learning algorithms. It includes methods for data preparation,
    visualization (contours and grids), Z-score analysis, and benchmarking.
    """

    def __init__(self):
        """
        Initializes the AnomalyDetector with a standard suite of algorithms.
        """
        self.algorithms = [
            ("Robust Covariance\n(Mahalanobis Dist.)", EllipticEnvelope(contamination=0.1)),
            ("Gaussian Mixture\n(Clustering + Mahalanobis)", GaussianMixture(n_components=2, covariance_type='full', random_state=42)),
            ("One-Class SVM", OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)),
            ("One-Class SVM (SGD)",
             make_pipeline(
                 Nystroem(gamma=0.1, random_state=42, n_components=150),
                 SGDOneClassSVM(
                     nu=0.15,
                     shuffle=True,
                     fit_intercept=True,
                     random_state=42,
                     tol=1e-6,
                 ),
             ),
             ),
            ("Isolation Forest", IsolationForest(contamination=0.1, random_state=42)),
            ("Local Outlier Factor", LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True))
            # novelty=True is required to use 'decision_function' for plotting contours
        ]

    def _generate_pca(self, df, n_components=2):
        """
        Helper to perform PCA reduction.
        """
        # 1. Select only numeric columns
        df_numeric = df.select_dtypes(include=[np.number])
        
        # 2. Fill NaNs
        X = df_numeric.fillna(0)
        
        # 3. Fit Transform
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        
        return X_pca, pca

    def plot_mahalanobis_contours(self, df, title="Mahalanobis Distance Analysis"):
        """
        Calculates and visualizes the Mahalanobis distance of samples in a reduced 2D PCA space.
        """
        # FIX 1: Select only numeric columns to avoid "could not convert string to float"
        df_numeric = df.select_dtypes(include=[np.number])
        
        # FIX 2: Fill NaNs in the numeric data
        X = df_numeric.fillna(0)
        
        # Determine PCA projection
        pca = PCA(n_components=2) # We only need 2 components for the 2D plot
        X_pca = pca.fit_transform(X)

        # 2. Fit Elliptic Envelope (Robust Covariance)
        ee = EllipticEnvelope(contamination=0.01, random_state=42)
        ee.fit(X_pca)

        # Prediction (-1 outlier, 1 inlier)
        y_pred = ee.predict(X_pca)

        # 3. Create Meshgrid for contour levels
        x_min, x_max = X_pca[:, 0].min() - 2, X_pca[:, 0].max() + 2
        y_min, y_max = X_pca[:, 1].min() - 2, X_pca[:, 1].max() + 2
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                             np.linspace(y_min, y_max, 500))

        # Calculate the decision function on the grid
        Z = ee.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # 4. Plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Draw contour levels
        ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu_r, alpha=0.2)
        ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')  # Outlier Boundary

        # Scatter Plot
        inliers = X_pca[y_pred == 1]
        outliers = X_pca[y_pred == -1]

        ax.scatter(inliers[:, 0], inliers[:, 1], c='blue', s=40, edgecolors='k', label='Normal Data (Inliers)')
        ax.scatter(outliers[:, 0], outliers[:, 1], c='red', s=60, marker='X', edgecolors='k', label='Outliers')

        ax.set_title(f"{title}\nRobust Mahalanobis Distance Contours", fontsize=14)
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.show()

    def plot_anomaly_grid(self, X, algorithms, fname: str, output_dir: str = '.', save_plots: bool = False):
        """
        Generates a grid of plots comparing different anomaly detection algorithms
        on the same dataset.

        Parameters:
        -----------
        X : array-like
            The input data (usually 2D projected).
        algorithms : list
            List of (name, model) tuples.
        fname : str
            Filename identifier.
        output_dir : str
            Directory to save the plot.
        save_plots : bool
            Whether to save the figure to disk.
        """
        # Setup plot grid (Adjusted for 6 algorithms: 3 rows x 2 columns)
        n_algos = len(algorithms)
        rows = (n_algos + 1) // 2
        fig, axs = plt.subplots(rows, 2, figsize=(12, 4 * rows))

        plt.subplots_adjust(left=.05, right=.95, bottom=.08, top=.92, wspace=.2, hspace=.35)
        axs = axs.flatten()

        # Meshgrid for contours
        x_min, x_max = X[:, 0].min() - 2, X[:, 0].max() + 2
        y_min, y_max = X[:, 1].min() - 2, X[:, 1].max() + 2
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))

        plot_num = 0

        for name, algorithm in algorithms:
            if plot_num >= len(axs): break
            ax = axs[plot_num]

            # Fit algorithm on the 2D projected data
            # Note: SGD requires positive values or specific scaling sometimes,
            # but StandardScaler usually works fine.
            try:
                algorithm.fit(X)

                # Logic for prediction and contours
                if "Gaussian Mixture" in name:
                    scores = algorithm.score_samples(X)
                    threshold = np.percentile(scores, 10)
                    y_pred = np.ones(X.shape[0])
                    y_pred[scores < threshold] = -1

                    Z = algorithm.score_samples(np.c_[xx.ravel(), yy.ravel()])
                else:
                    y_pred = algorithm.predict(X)

                    # Decision Function retrieval
                    if hasattr(algorithm, "decision_function"):
                        Z = algorithm.decision_function(np.c_[xx.ravel(), yy.ravel()])
                    elif hasattr(algorithm, "score_samples"):
                        Z = algorithm.score_samples(np.c_[xx.ravel(), yy.ravel()])
                    else:
                        # Fallback for models without decision_function (rare in this list)
                        Z = np.zeros(xx.ravel().shape)

                Z = Z.reshape(xx.shape)

                # --- VISUALIZATION ---
                # 1. Contours
                if "Gaussian Mixture" in name:
                    ax.contour(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 7),
                               linewidths=1.5, colors='darkgreen', alpha=0.6)
                else:
                    ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
                    ax.contour(xx, yy, Z, levels=np.linspace(Z.min(), 0, 5), linewidths=0.8, colors='black',
                               alpha=0.3)

                # 2. Scatter Plot
                colors = np.array(['gold' if x == -1 else '#4B0082' for x in y_pred])
                ax.scatter(X[:, 0], X[:, 1], s=25, c=colors, edgecolors='k', alpha=0.9, linewidth=0.5)

                ax.set_title(name, fontsize=11, fontweight='bold')
                ax.set_xticks(())
                ax.set_yticks(())

            except Exception as e:
                ax.text(0.5, 0.5, f"Fit Error:\n{str(e)}", ha='center')
                print(f"Error plotting {name}: {e}")

            plot_num += 1

        # Remove empty axes if any
        for i in range(plot_num, len(axs)):
            fig.delaxes(axs[i])

        # Main Title
        fig.suptitle(f"Anomaly Detection Comparison: {fname}", fontsize=14, fontweight='bold')

        # Legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Inliers (Normal)',
                   markerfacecolor='#4B0082', markersize=10, markeredgecolor='k'),
            Line2D([0], [0], marker='o', color='w', label='Outliers (Anomaly)',
                   markerfacecolor='gold', markersize=10, markeredgecolor='k'),
            Line2D([0], [0], color='black', lw=2, label='Decision Boundary')
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10, frameon=False)

        if save_plots:
            clean_name = os.path.splitext(fname)[0]
            save_path = os.path.join(output_dir, f"ANOMALY_COMPARISON_{clean_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()


    def calculate_z_scores(self, df):
        """
        Calculates Z-scores based on the sum of intensities per sample.
        This serves as a simple proxy for total metabolic content to identify
        overall intensity outliers.

        Parameters:
        -----------
        df : pd.DataFrame
            The input data.
        fname : str
            Dataset identifier.

        Returns:
        --------
        tuple
            (z_scores, std_dev)
        """           
        # Sum of intensities per sample (a simple proxy for total metabolic content)
        sample_sums = df.sum(axis=1)

        # Calculate Z-score: (x - mean) / std
        mean_val = sample_sums.mean()
        std_dev = sample_sums.std()

        z_scores = (sample_sums - mean_val) / std_dev

        return z_scores, std_dev
   
    def benchmark_algorithms(self, df, fname: str):
        """
        Benchmarks various anomaly detection algorithms using the Silhouette Score.
        Note: The algorithms used here are fresh instances fitted on the full
        dimensional data, not the 2D projection.
        
        Parameters:
        -----------
        df : pd.DataFrame
            The input data (scaled).
        fname : str
            Dataset identifier.

        Returns:
        --------
        pd.DataFrame
            A dataframe sorted by Silhouette Score containing benchmark results.
        """
        X_scaled = df

        # Define algorithms (Same configuration as above)
        # Note: We create new instances to avoid fitting on 2D data; benchmarking uses FULL DIMENSIONS
        algos = {
            "Isolation Forest": IsolationForest(contamination=0.1, random_state=42),
            "Robust Covariance": EllipticEnvelope(contamination=0.1),
            "One-Class SVM": OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1),
            "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, contamination=0.1),
            # novelty=False for fit_predict
        }

        results = []

        for name, model in algos.items():
            try:
                # Predict labels (-1 for outlier, 1 for inlier)
                if hasattr(model, 'fit_predict'):
                    labels = model.fit_predict(X_scaled)
                else:
                    model.fit(X_scaled)
                    labels = model.predict(X_scaled)

                # Check if we actually found outliers
                n_outliers = np.sum(labels == -1)

                if n_outliers > 0 and n_outliers < len(labels):
                    # Silhouette Score: Measures cluster separation (Inliers vs Outliers)
                    # Range: -1 to 1. Higher is better.
                    sil_score = silhouette_score(X_scaled, labels)
                    results.append({'Algorithm': name, 'Silhouette Score': sil_score, 'Outliers Found': n_outliers})
                else:
                    results.append({'Algorithm': name, 'Silhouette Score': -1.0, 'Outliers Found': n_outliers})

            except Exception as e:
                print(f"Benchmarking error for {name}: {e}")

        # Create comparison table
        bench_df = pd.DataFrame(results).sort_values(by='Silhouette Score', ascending=False)

        print(f"\n--- Anomaly Benchmark for {fname} ---")
        print(bench_df)
        return bench_df