import os

import pandas as pd
from matplotlib.patches import Ellipse
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedKFold, permutation_test_score
from sklearn.preprocessing import LabelEncoder

from config_visualization import *


def _calculate_vips(model):
    """
    Helper function to calculate Variable Importance in Projection (VIP) scores
    from a trained sklearn PLSRegression model.

    Formula reference:
    VIP_j = sqrt( p * sum( SSY_k * (w_jk / ||w_k||)^2 ) / sum(SSY) )
    """
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_

    p, h = w.shape

    vips = np.zeros((p,))

    # Sum of squares of explained variance of Y for each component
    s = np.diag(t.T @ t @ q.T @ q).reshape(h)
    total_s = np.sum(s)

    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)])
        vips[i] = np.sqrt(p * (s.T @ weight) / total_s)

    return vips


def run_pls_da(X_train, y_train, X_test, n_components=2, threshold=0.5):
    """
    Trains a PLS-DA (Partial Least Squares Discriminant Analysis) model.

    THEORY:
    - Algorithm: Uses NIPALS/SIMPLS to maximize covariance between X and Y.
    - Regression to Classification: Y is dummy-encoded (0/1). The model predicts a continuous score.
      A threshold (default 0.5) is applied to assign the class.
    - Feature Selection: Calculates VIP (Variable Importance in Projection) scores.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels (Categorical strings allowed).
        X_test (pd.DataFrame): Test features.
        n_components (int): Number of Latent Variables (LVs) to use.
                            MUST be optimized via Cross-Validation externally.
        threshold (float): Decision threshold for binary classification (default 0.5).

    Returns:
        tuple: (y_pred, y_prob, feature_info, model_results)
            - y_pred: Predicted class labels (original string labels).
            - y_prob: continuous regression output (can be treated as pseudo-probability).
            - feature_info: Dict containing VIP scores and Coefficients.
            - model_results: Dict containing 'model' (for plotting), 'x_scores', 'y_scores'.
    """

    # 1. Encode Labels (Strings -> 0/1) because PLS is a regressor
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    # Important: Check which class is 1 and which is 0 for interpretation
    # usually 0=First alphabetical, 1=Second.

    # 2. Initialize and Fit PLSRegression
    pls = PLSRegression(n_components=n_components, scale=False)  # Data assumed scaled externally
    pls.fit(X_train, y_train_enc)

    # 3. Prediction (Continuous Regression Output)
    y_pred_continuous = pls.predict(X_test).flatten()

    # 4. Classification (Thresholding)
    # Apply threshold to get 0/1
    y_pred_enc = (y_pred_continuous >= threshold).astype(int)
    # Convert back to original string labels (e.g., 'CHD', 'CTRL')
    y_pred = le.inverse_transform(y_pred_enc)

    # 5. Extract VIP Scores and Coefficients
    vip_scores = _calculate_vips(pls)
    coefficients = pls.coef_.flatten()

    feature_info = {
        'coefficients': coefficients,
        'vip_scores': vip_scores,
        'feature_importance_ranking': {}
    }

    if isinstance(X_train, pd.DataFrame):
        feature_names = X_train.columns
        # Map VIPs to names
        vip_dict = dict(zip(feature_names, vip_scores))
        # Sort by VIP (descending) -> Top Biomarkers
        sorted_vips = dict(sorted(vip_dict.items(), key=lambda item: item[1], reverse=True))
        feature_info['feature_importance_ranking'] = sorted_vips
    else:
        feature_info['feature_importance_ranking'] = dict(enumerate(vip_scores))

    # 6. Store Model Results for Visualization
    # We need the scores of the TRAINING set to plot the model structure
    # Scikit-learn stores them in x_scores_ and y_scores_
    model_results = {
        'model': pls,
        'train_x_scores': pd.DataFrame(pls.x_scores_, index=X_train.index,
                                       columns=[f'LV{i + 1}' for i in range(n_components)]),
        'train_y_enc': y_train_enc,  # Needed for coloring the score plot
        'class_names': le.classes_,  # Needed for legend
        'feature_names': X_train.columns if isinstance(X_train, pd.DataFrame) else range(X_train.shape[1])
    }

    # Return structure matches other models, plus the model_results for plotting
    return y_pred, y_pred_continuous, feature_info, model_results

def optimize_pls_components(X, y, max_components=10, cv_splits=5):
    """
    Evaluates PLS-DA performance across different numbers of Latent Variables (LVs)
    to find the optimal complexity and avoid overfitting.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Labels.
        max_components (int): Max number of LVs to test.
        cv_splits (int): Number of Cross-Validation folds.

    Returns:
        pd.DataFrame: Performance metrics for each number of components.
    """
    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    results = []

    # Define CV strategy (Stratified to keep class balance)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    print(f"Optimizing PLS-DA components (Testing 1 to {max_components} LVs)...")

    for n in range(1, max_components + 1):
        # Create PLS model (Regression mode)
        # Note: sklearn PLS does not support 'scoring' for classification directly in cross_val_score
        # without a wrapper, but R2 is a decent proxy for regression fit,
        # or we manually calculate accuracy. Let's do manual loop for Accuracy.

        fold_accuracies = []

        for train_idx, val_idx in cv.split(X, y_enc):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y_enc[train_idx], y_enc[val_idx]

            # Train PLS
            pls = PLSRegression(n_components=n, scale=False)
            pls.fit(X_train_fold, y_train_fold)

            # Predict (Threshold 0.5)
            preds = (pls.predict(X_val_fold).flatten() >= 0.5).astype(int)

            # Calculate Accuracy
            acc = np.mean(preds == y_val_fold)
            fold_accuracies.append(acc)

        mean_acc = np.mean(fold_accuracies)
        results.append({'n_components': n, 'Accuracy': mean_acc, 'Error': 1 - mean_acc})

    return pd.DataFrame(results)


def run_permutation_test(X, y, n_components=2, n_permutations=100):
    """
    Performs a Permutation Test to validate the model's statistical significance.

    Args:
        X, y: Data and Labels.
        n_components: The optimal number of LVs chosen.
        n_permutations: How many times to shuffle labels.

    Returns:
        tuple: (real_score, permutation_scores, p_value)
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # We need a custom scorer because PLS is a regressor used as classifier
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

def plot_pls_scores(model_results, output_dir, lv_x=1, lv_y=2, file_name="pls_score_plot", show_ellipse=True):
    """
    Generates the PLS-DA Score Plot (Latent Variable 1 vs Latent Variable 2).

    THEORY:
    - Shows how samples separate based on MAXIMIZING COVARIANCE with the class.
    - Supervised separation: Clusters should be tighter/more distinct than PCA.
    """
    os.makedirs(output_dir, exist_ok=True)

    scores = model_results['train_x_scores']
    y_enc = model_results['train_y_enc']
    class_names = model_results['class_names']

    # Recover original class labels from encoding for the legend
    labels = [class_names[val] for val in y_enc]

    lv_x_label = f"LV{lv_x}"
    lv_y_label = f"LV{lv_y}"

    x_data = scores[lv_x_label]
    y_data = scores[lv_y_label]

    # Setup Palette
    unique_classes = np.unique(labels)
    current_palette = DISCRETE_COLORS[:len(unique_classes)]
    markers_dict = {cls: MARKERS[i % len(MARKERS)] for i, cls in enumerate(unique_classes)}

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.scatterplot(x=x_data, y=y_data, hue=labels, style=labels,
                    palette=current_palette, markers=markers_dict,
                    s=100, alpha=0.9, edgecolor='k', ax=ax)

    # --- Ellipse (Optional) ---
    if show_ellipse:
        # Note: In PLS-DA, ellipses are often drawn per-class, but a global T2
        # is also valid to spot outliers relative to the overall model center.
        chi2_val = 5.991
        std_x = np.std(x_data)
        std_y = np.std(y_data)
        width = 2 * np.sqrt(chi2_val) * std_x
        height = 2 * np.sqrt(chi2_val) * std_y
        ellipse = Ellipse((0, 0), width=width, height=height,
                          facecolor='none', edgecolor='black', linestyle='--', linewidth=1.5)
        ax.add_patch(ellipse)

    ax.set_xlabel(f"{lv_x_label} (Latent Variable)", fontweight='bold')
    ax.set_ylabel(f"{lv_y_label} (Latent Variable)", fontweight='bold')
    ax.set_title(f"PLS-DA Score Plot: {lv_x_label} vs {lv_y_label}", fontweight='bold')

    ax.axhline(0, color='grey', linestyle=':', linewidth=0.8)
    ax.axvline(0, color='grey', linestyle=':', linewidth=0.8)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{file_name}_LV{lv_x}vsLV{lv_y}.{SAVE_FORMAT}")
    plt.savefig(save_path, format=SAVE_FORMAT, bbox_inches='tight')
    plt.close(fig)
    print(f"Generating PLS-DA Score Plot saved to {save_path}")


def plot_vip_scores(feature_info, output_dir, file_name="vip_scores_plot", top_n=20, vip_threshold=1.0):
    """
    Generates a Variable Importance in Projection (VIP) plot.

    THEORY:
    - VIP > 1.0 (or 0.8) indicates highly influential variables.
    - Used for Feature Selection in PLS-DA.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract data
    ranking = feature_info['feature_importance_ranking']

    # Convert to DataFrame for plotting
    df_vip = pd.DataFrame(list(ranking.items()), columns=['Feature', 'VIP'])

    # Filter Top N
    df_plot = df_vip.head(top_n)

    fig, ax = plt.subplots(figsize=(8, 10))  # Tall plot for names

    # Color logic: Highlight bars above threshold
    # Using DISCRETE_COLORS[0] (e.g. Purple) for High, Grey for Low
    colors = [DISCRETE_COLORS[0] if x >= vip_threshold else 'lightgrey' for x in df_plot['VIP']]

    sns.barplot(data=df_plot, x='VIP', y='Feature', palette=colors, ax=ax, edgecolor='black')

    # Threshold Line
    ax.axvline(vip_threshold, color='red', linestyle='--', linewidth=1.5, label=f'VIP Threshold ({vip_threshold})')

    ax.set_xlabel("VIP Score")
    ax.set_ylabel("Metabolite")
    ax.set_title(f"Top {top_n} Features by VIP Score", fontweight='bold')
    ax.legend(loc='lower right')

    ax.grid(True, axis='x', linestyle='--', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{file_name}.{SAVE_FORMAT}")
    plt.savefig(save_path, format=SAVE_FORMAT, bbox_inches='tight')
    plt.close(fig)
    print(f"Generating VIP Scores Plot saved to {save_path}")


def plot_pls_classification_error(optimization_df, output_dir, file_name="pls_optimization"):
    """
    Plots Accuracy/Error vs Number of Components to select optimal LVs.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    x = optimization_df['n_components']
    y_acc = optimization_df['Accuracy']

    # Plot Accuracy
    ax.plot(x, y_acc, marker='o', color=DISCRETE_COLORS[0], linewidth=2, label='CV Accuracy')

    # Mark the max
    best_n = optimization_df.loc[optimization_df['Accuracy'].idxmax(), 'n_components']
    best_acc = optimization_df['Accuracy'].max()
    ax.axvline(best_n, color='red', linestyle='--', alpha=0.6, label=f'Optimal LVs ({int(best_n)})')

    ax.set_title("PLS-DA Component Optimization (Cross-Validation)", fontweight='bold')
    ax.set_xlabel("Number of Latent Variables (LVs)")
    ax.set_ylabel("Classification Accuracy")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{file_name}.{SAVE_FORMAT}")
    plt.savefig(save_path, format=SAVE_FORMAT, bbox_inches='tight')
    plt.close(fig)
    print(f"Generating PLS Optimization Plot saved to {save_path}")


def plot_permutation_test(real_score, perm_scores, p_value, output_dir, file_name="pls_permutation_test"):
    """
    Visualizes the Permutation Test results (Histogram).
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Histogram of random scores
    sns.histplot(perm_scores, bins=20, color='lightgrey', edgecolor='gray', label='Random Permutations', ax=ax)

    # Vertical line for real score
    ax.axvline(real_score, color='red', linestyle='--', linewidth=2, label=f'Real Model (Acc={real_score:.2f})')

    ax.set_title(f"Permutation Test Validation (p-value = {p_value:.4f})", fontweight='bold')
    ax.set_xlabel("Accuracy Score")
    ax.set_ylabel("Frequency")
    ax.legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{file_name}.{SAVE_FORMAT}")
    plt.savefig(save_path, format=SAVE_FORMAT, bbox_inches='tight')
    plt.close(fig)
    print(f"Generating Permutation Test Plot saved to {save_path}")


def plot_pls_predicted_vs_observed(y_true, y_pred_continuous, output_dir, file_name="pls_pred_vs_obs"):
    """
    Scatter plot of Predicted (Continuous) vs Observed (Categorical) values.
    Shows the decision threshold.
    """
    os.makedirs(output_dir, exist_ok=True)

    # y_true are strings (e.g. CTRL, CHD). Convert to 0/1 for plotting on Y-axis
    # But wait! The plot usually has Sample Index on X and Predicted Value on Y

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create indices
    indices = np.arange(len(y_true))

    # Color based on TRUE class
    unique_classes = np.unique(y_true)
    # Fix Warning: Slice palette
    current_palette = DISCRETE_COLORS[:len(unique_classes)]

    # We plot: X=Index, Y=Predicted Value, Color=True Class
    sns.scatterplot(x=indices, y=y_pred_continuous, hue=y_true, palette=current_palette,
                    s=80, edgecolor='k', ax=ax)

    # Threshold line
    ax.axhline(0.5, color='red', linestyle='--', label='Decision Threshold (0.5)')

    # Shaded regions for classes
    ax.axhspan(0.5, max(y_pred_continuous.max(), 1.1), color=current_palette[1], alpha=0.1)  # Predicted Class 1 area
    ax.axhspan(min(y_pred_continuous.min(), -0.1), 0.5, color=current_palette[0], alpha=0.1)  # Predicted Class 0 area

    ax.set_title("PLS-DA: Predicted vs Observed", fontweight='bold')
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Predicted Score (Regression Output)")
    ax.legend(loc='lower right')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{file_name}.{SAVE_FORMAT}")
    plt.savefig(save_path, format=SAVE_FORMAT, bbox_inches='tight')
    plt.close(fig)
    print(f"Generating Predicted vs Observed Plot saved to {save_path}")


def plot_pls_loadings(model_results, output_dir, lv_x=1, lv_y=2, file_name="pls_loadings_plot", top_n=20):
    """
    Generates PLS Loadings Scatter Plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    model = model_results['model']
    feature_names = model_results['feature_names']

    # Extract Loadings (x_loadings_)
    loadings_mat = model.x_loadings_

    # Create DataFrame
    cols = [f"LV{i + 1}" for i in range(loadings_mat.shape[1])]
    loadings_df = pd.DataFrame(loadings_mat, index=feature_names, columns=cols)

    lv_x_label = f"LV{lv_x}"
    lv_y_label = f"LV{lv_y}"

    # Reuse logic from PCA Loadings Plot logic
    magnitude = np.sqrt(loadings_df[lv_x_label] ** 2 + loadings_df[lv_y_label] ** 2)
    loadings_df['magnitude'] = magnitude
    top_loadings = loadings_df.sort_values(by='magnitude', ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Background points
    ax.scatter(loadings_df[lv_x_label], loadings_df[lv_y_label], color=DISCRETE_COLORS[3], alpha=0.4, s=20,
               label='All Features')

    # Top points
    ax.scatter(top_loadings[lv_x_label], top_loadings[lv_y_label], color=DISCRETE_COLORS[0], s=50,
               label=f'Top {top_n} Contributors')

    # Annotate
    texts = []
    for feature, row in top_loadings.iterrows():
        texts.append(plt.text(row[lv_x_label], row[lv_y_label], feature, fontsize=8, color=DISCRETE_COLORS[0],
                              fontweight='bold'))

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
    print(f"Generating PLS Loadings Plot saved to {save_path}")