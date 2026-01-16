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

from src.visualization import save_plot

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



    def plot_mahalanobis_contours(self, df, filename, output_dir):
        """
        Calculates and visualizes the Mahalanobis distance of samples in a reduced 2D PCA space.
        """
        # FIX 1: Select only numeric columns
        df_numeric = df.select_dtypes(include=[np.number])
        
        # FIX 2: Fill NaNs
        X = df_numeric.fillna(0)
        
        # Determine PCA projection
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        # Calculate Explained Variance
        exp_var_pc1 = pca.explained_variance_ratio_[0] * 100
        exp_var_pc2 = pca.explained_variance_ratio_[1] * 100

        # 2. Fit Elliptic Envelope (Robust Covariance)
        ee = EllipticEnvelope(contamination=0.01, random_state=42)
        ee.fit(X_pca)
        y_pred = ee.predict(X_pca)

        # --- FIX 3: DYNAMIC PADDING ---
        # Calculate range (max - min) for each axis
        x_range = X_pca[:, 0].max() - X_pca[:, 0].min()
        y_range = X_pca[:, 1].max() - X_pca[:, 1].min()

        # Define padding as 10% of the range (prevents "zoomed in" look)
        x_pad = x_range * 0.1
        y_pad = y_range * 0.1

        # Apply padding to limits
        x_min, x_max = X_pca[:, 0].min() - x_pad, X_pca[:, 0].max() + x_pad
        y_min, y_max = X_pca[:, 1].min() - y_pad, X_pca[:, 1].max() + y_pad
        # ------------------------------

        # 3. Create Meshgrid
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                            np.linspace(y_min, y_max, 500))

        # Calculate decision function
        Z = ee.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # 4. Plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Draw contour levels
        ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu_r, alpha=0.2)
        ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')

        # Scatter Plot
        inliers = X_pca[y_pred == 1]
        outliers = X_pca[y_pred == -1]

        ax.scatter(inliers[:, 0], inliers[:, 1], c='blue', s=40, edgecolors='k', label='Normal Data (Inliers)')
        ax.scatter(outliers[:, 0], outliers[:, 1], c='red', s=60, marker='X', edgecolors='k', label='Outliers')

        ax.set_title(f"{filename}", fontsize=14)
        ax.set_xlabel(f'PC1 ({exp_var_pc1:.2f}%)', fontsize=12)
        ax.set_ylabel(f'PC2 ({exp_var_pc2:.2f}%)', fontsize=12)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)


        save_plot(plt, filename, output_dir)
        #plt.show()

    def plot_anomaly_grid(self, X, filename, output_dir, algorithms):
        """
        Generates a grid of plots comparing different anomaly detection algorithms
        on the same dataset.
        """

        # Setup plot grid (Adjusted for 6 algorithms: 3 rows x 2 columns)
        n_algos = len(algorithms)
        rows = (n_algos + 1) // 2
        fig, axs = plt.subplots(rows, 2, figsize=(12, 4 * rows))

        plt.subplots_adjust(left=.05, right=.95, bottom=.08, top=.92, wspace=.2, hspace=.35)
        axs = axs.flatten()

        # --- FIX 1: DYNAMIC PADDING ---
        # Calculate proper padding (e.g., 20% of the range) to ensure boundaries are visible
        x_range = X[:, 0].max() - X[:, 0].min()
        y_range = X[:, 1].max() - X[:, 1].min()
        
        x_pad = x_range * 0.2
        y_pad = y_range * 0.2

        x_min, x_max = X[:, 0].min() - x_pad, X[:, 0].max() + x_pad
        y_min, y_max = X[:, 1].min() - y_pad, X[:, 1].max() + y_pad
        
        # Create larger meshgrid
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        # ------------------------------

        plot_num = 0

        for name, algorithm in algorithms:
            if plot_num >= len(axs): break
            ax = axs[plot_num]

            try:
                algorithm.fit(X)

                # Logic for prediction and contours
                if "Gaussian Mixture" in name:
                    scores = algorithm.score_samples(X)
                    # Threshold: lowest 10%
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
                        Z = np.zeros(xx.ravel().shape)

                Z = Z.reshape(xx.shape)

                # --- VISUALIZATION ---
                # 1. Contours
                if "Gaussian Mixture" in name:
                    # GMM Contours based on density
                    ax.contour(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 7),
                               linewidths=1.5, colors='darkgreen', alpha=0.6)
                else:
                    # FIX 2: Check if 0 is within Z range before plotting to avoid warnings
                    if Z.min() < 0 < Z.max():
                        ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
                    
                    # Fill contours (using safe linspace)
                    # We ensure the end is at least slightly above min to avoid range errors
                    end_level = min(0, Z.max())
                    if Z.min() < end_level:
                        levels = np.linspace(Z.min(), end_level, 5)
                        ax.contour(xx, yy, Z, levels=levels, linewidths=0.8, colors='black', alpha=0.3)

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
        fig.suptitle(f"Anomaly Detection Comparison: {filename}", fontsize=14, fontweight='bold')

        # Legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Inliers (Normal)',
                   markerfacecolor='#4B0082', markersize=10, markeredgecolor='k'),
            Line2D([0], [0], marker='o', color='w', label='Outliers (Anomaly)',
                   markerfacecolor='gold', markersize=10, markeredgecolor='k'),
            Line2D([0], [0], color='black', lw=2, label='Decision Boundary')
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10, frameon=False)

        save_plot(plt, filename, output_dir)


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
        # FIX: Added numeric_only=True to prevent errors with string columns (e.g., IDs)
        sample_sums = df.sum(axis=1, numeric_only=True)

        # Calculate Z-score: (x - mean) / std
        mean_val = sample_sums.mean()
        std_dev = sample_sums.std() # This calculates std of raw intensities

        z_scores = (sample_sums - mean_val) / std_dev

        return z_scores, std_dev
   

   
    def benchmark_algorithms(self, df, fname: str):
        """
        Benchmarks various anomaly detection algorithms using the Silhouette Score.
        """
        # FIX 1: Select numeric columns AND convert to numpy array immediately (.values)
        # This prevents the "fitted with feature names" warning in LOF
        X_scaled = df.select_dtypes(include=[np.number]).values

        # Define algorithms
        algos = {
            "Robust Covariance": EllipticEnvelope(contamination=0.1),
            
            # FIX 2: Added reg_covar=1e-5 to prevent "ill-defined empirical covariance" crash
            "Gaussian Mixture": GaussianMixture(n_components=2, covariance_type='full', 
                                                random_state=42, reg_covar=1e-5),
            
            "One-Class SVM": OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1),
            
            "One-Class SVM (SGD)": make_pipeline(
                Nystroem(gamma=0.1, random_state=42, n_components=150),
                SGDOneClassSVM(nu=0.15, shuffle=True, fit_intercept=True, random_state=42, tol=1e-6)
            ),
            
            "Isolation Forest": IsolationForest(contamination=0.1, random_state=42),
            
            "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
        }

        results = []

        print(f"\n--- Running Benchmark for {fname} ---")

        for name, model in algos.items():
            try:
                # --- Special Handling for Gaussian Mixture ---
                if name == "Gaussian Mixture":
                    model.fit(X_scaled)
                    # GMM doesn't predict -1/1. We use log-likelihood (score_samples).
                    scores = model.score_samples(X_scaled)
                    # Threshold: Mark the lowest 10% density samples as outliers
                    threshold = np.percentile(scores, 10) 
                    labels = np.where(scores < threshold, -1, 1)

                # --- Handling for Standard Detectors ---
                else:
                    # Fit and Predict
                    if hasattr(model, 'fit_predict') and not (name == "Local Outlier Factor"):
                        labels = model.fit_predict(X_scaled)
                    else:
                        # LOF with novelty=True or Pipeline requires fit() then predict()
                        model.fit(X_scaled)
                        labels = model.predict(X_scaled)

                # --- Calculate Metrics ---
                # Check if we actually found outliers
                n_outliers = np.sum(labels == -1)

                if n_outliers > 0 and n_outliers < len(labels):
                    # Silhouette Score: Measures cluster separation
                    sil_score = silhouette_score(X_scaled, labels)
                    results.append({'Algorithm': name, 'Silhouette Score': sil_score, 'Outliers Found': n_outliers})
                else:
                    results.append({'Algorithm': name, 'Silhouette Score': -1.0, 'Outliers Found': n_outliers})

            except Exception as e:
                print(f"Benchmarking error for {name}: {e}")

        if not results:
            print("No valid benchmark results generated.")
            return pd.DataFrame()

        # Create comparison table
        bench_df = pd.DataFrame(results).sort_values(by='Silhouette Score', ascending=False)

        print(bench_df)
        return bench_df


    def remove_outliers(self, df, samples_to_remove):
        """
        Removes samples from the dataframe based on a provided container of IDs.
        Handles lists, pandas Indices, or pandas Series (extracts the index automatically).

        Parameters:
        -----------
        df : pd.DataFrame
            The input dataframe.
        samples_to_remove : list, pd.Index, or pd.Series
            The samples to remove. If a Series is passed (e.g., outliers_z), 
            its index is used as the list of IDs.

        Returns:
        --------
        pd.DataFrame
            The cleaned dataframe.
        """
        # 1. Handle "None" or empty inputs safely
        if samples_to_remove is None or len(samples_to_remove) == 0:
            print("\nNo samples provided for removal. Returning original dataframe.")
            return df

        # 2. TYPE CHECKING: Extract IDs based on input type
        if isinstance(samples_to_remove, pd.Series):
            # If it's a Series (like your Z-score output), we want the INDEX (Sample IDs)
            print("  -> Input is a Series; extracting indices...")
            target_ids = samples_to_remove.index.tolist()
        
        elif isinstance(samples_to_remove, pd.Index):
            # If it's a raw pandas Index
            target_ids = samples_to_remove.tolist()
            
        elif isinstance(samples_to_remove, list):
            # If it's already a list
            target_ids = samples_to_remove
            
        else:
            # Fallback for numpy arrays or other iterables
            target_ids = list(samples_to_remove)

        # 3. Filter IDs: Only remove samples that actually exist in this dataframe
        valid_samples_to_drop = [s for s in target_ids if s in df.index]

        # 4. Perform Removal
        if valid_samples_to_drop:
            print(f"\n--- Removing {len(valid_samples_to_drop)} Outliers ---")
            df_cleaned = df.drop(index=valid_samples_to_drop)
            print(f"Data shape before: {df.shape} -> after: {df_cleaned.shape}")
            return df_cleaned
        else:
            print("\nNone of the provided outlier IDs were found in this dataframe.")
            return df

    def identify_consensus_outliers(self, df, fname: str):
        """
        Identifies outliers based on the consensus of the top 3 performing algorithms:
        Local Outlier Factor, Isolation Forest, and Robust Covariance.

        Parameters:
        -----------
        df : pd.DataFrame
            The input dataframe (samples as rows).
        fname : str
            Dataset identifier.

        Returns:
        --------
        list
            A list of sample indices (names) that are considered outliers by ALL 3 models.
        """
        print(f"\n--- Calculating Consensus Outliers for {fname} ---")
        
        # 1. Prepare Data (Numeric only, convert to numpy)
        X = df.select_dtypes(include=[np.number]).values
        sample_indices = df.index  # Keep track of sample names

        # 2. Define the Top 3 Algorithms
        # (Using the same settings that worked well in the benchmark)
        detectors = {
            "LOF": LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=False), # novelty=False for fit_predict
            "IsoForest": IsolationForest(contamination=0.1, random_state=42),
            "RobustCov": EllipticEnvelope(contamination=0.1, random_state=42)
        }

        outlier_votes = pd.DataFrame(index=sample_indices)

        # 3. Run each model and collect votes
        for name, model in detectors.items():
            try:
                if name == "LOF":
                    # LOF fit_predict (standard mode)
                    labels = model.fit_predict(X)
                else:
                    # Others
                    labels = model.fit_predict(X)
                
                # Store result: True if outlier (-1), False if inlier (1)
                outlier_votes[name] = (labels == -1)
                
            except Exception as e:
                print(f"Error running {name} for consensus: {e}")

        # 4. Find Consensus (Intersection)
        # A sample is a consensus outlier if ALL columns are True
        outlier_votes['is_consensus'] = outlier_votes.all(axis=1)
        
        consensus_outliers = outlier_votes[outlier_votes['is_consensus']].index.tolist()

        # 5. Print Summary
        print(f"Total Samples: {len(df)}")
        print(f"Consensus Outliers Found (Agreed by LOF, IsoForest, RobustCov): {len(consensus_outliers)}")
        print(f"Outlier Sample IDs: {consensus_outliers}")

        return consensus_outliers