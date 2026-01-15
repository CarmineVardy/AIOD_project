import os

import pandas as pd
from scipy.stats import f
from sklearn.decomposition import PCA

from config_visualization import *


class SIMCAModel:
    """
    Soft Independent Modeling of Class Analogy (SIMCA).

    THEORY & IMPLEMENTATION:
    ------------------------
    1. Independent PCA Models: Fits a separate PCA for each class in the training data.
    2. Soft Classification: Calculates Hotelling's T2 and Q-Residuals (SPE) for new samples
       against each class model.
    3. Decision Rule: A sample is assigned to a class if its combined distance D
       is within the critical limit (sqrt(2) or 1).

    Can result in:
    - Single Class Membership (Ideal)
    - Multiple Class Membership ("Confused")
    - No Class Membership ("Alien" / Outlier)
    """

    def __init__(self, variance_threshold=0.95):
        """
        Args:
            variance_threshold (float): Variance to explain to automatically select # PCs per class.
        """
        self.variance_threshold = variance_threshold
        self.class_models = {}  # Stores PCA model + stats for each class
        self.classes_ = []

    def fit(self, X, y):
        """
        Trains independent PCA models for each class found in y.
        """
        # Ensure input is standard pandas/numpy
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        y = pd.Series(y) if not isinstance(y, pd.Series) else y

        self.classes_ = y.unique()

        for cls in self.classes_:
            # 1. Subset Data
            X_cls = X[y == cls]

            # 2. Fit PCA
            # We use enough components to explain variance_threshold
            pca = PCA(n_components=self.variance_threshold)
            pca.fit(X_cls)

            # 3. Calculate Critical Limits (Training Phase)
            # T2 Limit (Hotelling)
            n_samples, n_features = X_cls.shape
            n_components = pca.n_components_

            # F-distribution critical value for T2
            # Alpha = 0.05 (95% confidence)
            F_crit = f.ppf(0.95, n_components, n_samples - n_components)
            t2_limit = (n_components * (n_samples ** 2 - 1)) / (n_samples * (n_samples - n_components)) * F_crit

            # Q-Residual Limit (Approximation using Chi-Square or empirical)
            # Simple approach: 95th percentile of training Q-residuals
            X_pca = pca.transform(X_cls)
            X_reconstructed = pca.inverse_transform(X_pca)
            residuals = X_cls - X_reconstructed
            q_values = np.sum(residuals ** 2, axis=1)
            q_limit = np.percentile(q_values, 95)

            # Store Model & Limits
            self.class_models[cls] = {
                'pca': pca,
                't2_limit': t2_limit,
                'q_limit': q_limit,
                'n_components': n_components
            }

    def predict(self, X):
        """
        Predicts class labels based on combined distance.
        Returns 'Confused' if >1 class accepted, 'Alien' if 0 accepted.
        """
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        predictions = []

        for _, sample in X.iterrows():
            accepted_classes = []

            # Test against EACH class model independently
            for cls, model_info in self.class_models.items():
                pca = model_info['pca']
                t2_lim = model_info['t2_limit']
                q_lim = model_info['q_limit']

                # Reshape for sklearn
                sample_reshaped = sample.values.reshape(1, -1)

                # 1. Calculate T2 (Inner Distance)
                score = pca.transform(sample_reshaped)[0]
                # T2 formula: sum(score^2 / eigenvalue)
                eigenvalues = pca.explained_variance_
                t2 = np.sum((score ** 2) / eigenvalues)

                # 2. Calculate Q (Outer/Residual Distance)
                reconstruction = pca.inverse_transform(score.reshape(1, -1))
                residual = sample_reshaped - reconstruction
                q = np.sum(residual ** 2)

                # 3. Combined Distance (Normalized)
                # D = sqrt( (Q/Q_lim)^2 + (T2/T2_lim)^2 )
                # Often simplified or summed. Let's use the combined Euclidean ratio.
                # If d_combined <= sqrt(2), sample is inside the acceptance area.
                d_combined = np.sqrt((q / q_lim) ** 2 + (t2 / t2_lim) ** 2)

                if d_combined <= np.sqrt(2):
                    accepted_classes.append(cls)

            # 4. Final Assignment Logic
            if len(accepted_classes) == 1:
                predictions.append(accepted_classes[0])
            elif len(accepted_classes) > 1:
                predictions.append('Confused')
            else:
                predictions.append('Alien')

        return np.array(predictions)

    def get_distances_for_plot(self, X):
        """
        Calculates normalized distances for Coomans' Plot.
        Returns a dict: {Class_A: [distances], Class_B: [distances]}
        """
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        results = {}

        for cls, model_info in self.class_models.items():
            pca = model_info['pca']
            t2_lim = model_info['t2_limit']
            q_lim = model_info['q_limit']

            distances = []
            for _, sample in X.iterrows():
                sample_reshaped = sample.values.reshape(1, -1)

                score = pca.transform(sample_reshaped)[0]
                eigenvalues = pca.explained_variance_
                t2 = np.sum((score ** 2) / eigenvalues)

                reconstruction = pca.inverse_transform(score.reshape(1, -1))
                residual = sample_reshaped - reconstruction
                q = np.sum(residual ** 2)

                # Normalized combined distance
                d_norm = np.sqrt((q / q_lim) ** 2 + (t2 / t2_lim) ** 2)
                distances.append(d_norm)

            results[cls] = np.array(distances)

        return results


def run_simca(X_train, y_train, X_test, variance_threshold=0.95):
    """
    Wrapper to train and run SIMCA model.
    """
    model = SIMCAModel(variance_threshold=variance_threshold)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # For SIMCA, probabilities are not straightforward (it's distance-based).
    # We return None for y_prob.
    y_prob = None

    # Store distance info for Coomans' plot
    plot_distances = model.get_distances_for_plot(X_test)

    feature_info = {
        'model': model,  # Passing full object to access internal PCAs if needed
        'plot_distances': plot_distances
    }

    return y_pred, y_prob, feature_info

def plot_simca_coomans(feature_info, y_true, output_dir, file_name="simca_coomans_plot"):
    """
    Generates Coomans' Plot for SIMCA classification.

    THEORY:
    - X-axis: Distance to Class 1 Model.
    - Y-axis: Distance to Class 2 Model.
    - Critical Limit: sqrt(2) (normalized distance). Samples below this are accepted.
    - Quadrants:
        - Bottom-Left: "Confused" (Accepted by both).
        - Top-Right: "Alien" (Rejected by both).
        - Top-Left: Class 2 Only.
        - Bottom-Right: Class 1 Only.
    """
    os.makedirs(output_dir, exist_ok=True)

    distances = feature_info['plot_distances']

    # Ensure we have exactly 2 classes for a 2D plot
    classes = list(distances.keys())
    if len(classes) != 2:
        print("Warning: Coomans' plot is designed for 2 classes comparison. Skipping.")
        return

    cls1, cls2 = classes[0], classes[1]

    dist_1 = distances[cls1]
    dist_2 = distances[cls2]

    # Critical Limit (Normalized Distance threshold is usually sqrt(2) ~ 1.41)
    critical_limit = np.sqrt(2)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Setup Palette for TRUE labels
    unique_true_classes = np.unique(y_true)
    # Fix Warning: Slice palette
    current_palette = DISCRETE_COLORS[:len(unique_true_classes)]
    markers_dict = {cls: MARKERS[i % len(MARKERS)] for i, cls in enumerate(unique_true_classes)}

    # Scatter Plot
    sns.scatterplot(x=dist_1, y=dist_2, hue=y_true, style=y_true,
                    palette=current_palette, markers=markers_dict,
                    s=100, alpha=0.9, edgecolor='k', ax=ax)

    # Draw Critical Limits (Lines)
    ax.axvline(critical_limit, color='red', linestyle='--', linewidth=1.5, label='Critical Limit')
    ax.axhline(critical_limit, color='red', linestyle='--', linewidth=1.5)

    # Label Quadrants
    # Find max limits to place text roughly in corners
    max_x = max(dist_1.max(), critical_limit * 1.5)
    max_y = max(dist_2.max(), critical_limit * 1.5)

    ax.text(critical_limit / 2, critical_limit / 2, "CONFUSED", color='grey', ha='center', va='center',
            fontweight='bold', alpha=0.5)
    ax.text(max_x * 0.9, max_y * 0.9, "ALIEN", color='grey', ha='center', va='center', fontweight='bold', alpha=0.5)
    ax.text(max_x * 0.9, critical_limit / 2, f"{cls1} MEMBERS", color=current_palette[0], ha='center', va='center',
            fontweight='bold', alpha=0.5)
    ax.text(critical_limit / 2, max_y * 0.9, f"{cls2} MEMBERS", color=current_palette[1], ha='center', va='center',
            fontweight='bold', alpha=0.5)

    ax.set_xlabel(f"Normalized Distance to {cls1} Model", fontweight='bold')
    ax.set_ylabel(f"Normalized Distance to {cls2} Model", fontweight='bold')
    ax.set_title(f"SIMCA Coomans' Plot ({cls1} vs {cls2})", fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{file_name}.{SAVE_FORMAT}")
    plt.savefig(save_path, format=SAVE_FORMAT, bbox_inches='tight')
    plt.close(fig)
    print(f"Generating SIMCA Coomans' Plot saved to {save_path}")