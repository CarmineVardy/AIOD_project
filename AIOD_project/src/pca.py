import os

import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config_visualization import *


def perform_pca(df, n_components=None, scaling='autoscaling'):
    """
    Performs Principal Component Analysis (PCA) on the provided dataset.

    THEORY & IMPLEMENTATION DETAILS:
    --------------------------------
    1. Preprocessing:
       - PCA is scale-dependent.
       - Options provided: 'autoscaling' (mean=0, std=1) or 'pareto' (mean=0, std=sqrt(std)).
       - User must ensure data is cleaned (no NaNs) before calling this.
    2. Calculation:
       - Uses SVD decomposition via sklearn.
    3. Output Organization:
       - Returns a dictionary containing Scores (Samples), Loadings (Features),
         and Variance stats, all labeled with proper indices.

    Args:
        df (pd.DataFrame): Input data (Samples x Features).
        n_components (int): Number of components to keep. If None, keeps all.
        scaling (str): 'autoscaling', 'pareto', or None.
                       - 'autoscaling': Recommended for general LC-MS.
                       - 'pareto': Recommended if noise is high (reduces noise amplification).
                       - None: Assumes data is already scaled externally.

    Returns:
        dict: A dictionary containing:
            - 'model': The trained PCA sklearn object.
            - 'scores': pd.DataFrame of Score values (Samples x PCs).
            - 'loadings': pd.DataFrame of Loading values (Features x PCs).
            - 'explained_variance': Array of variance ratio per PC.
            - 'cumulative_variance': Array of cumulative variance.
    """
    # Separate numeric data for PCA calculation
    df = df.drop(columns=['Class'])

    # 1. Scaling / Preprocessing
    data_mat = df.values

    if scaling == 'autoscaling':
        # Mean=0, Std=1
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_mat)
    elif scaling == 'pareto':
        # Mean Centering
        data_centered = data_mat - np.mean(data_mat, axis=0)
        # Pareto: Divide by sqrt(STD)
        data_scaled = data_centered / np.sqrt(np.std(data_mat, axis=0))
    else:
        # Assume external scaling or just Mean Centering (default in PCA)
        data_scaled = data_mat

    # 2. PCA Execution
    # If n_components is None, sklearn computes all min(n_samples, n_features)
    pca = PCA(n_components=n_components)
    scores_data = pca.fit_transform(data_scaled)

    # 3. Formatting Outputs
    # Create column names [PC1, PC2, ..., PCn]
    pc_labels = [f"PC{i + 1}" for i in range(scores_data.shape[1])]

    # Scores DataFrame (Rows=Samples)
    scores_df = pd.DataFrame(data=scores_data, index=df.index, columns=pc_labels)

    # Loadings DataFrame (Rows=Features)
    # sklearn components_ is (n_components, n_features), so we transpose
    loadings_df = pd.DataFrame(data=pca.components_.T, index=df.columns, columns=pc_labels)

    results = {
        'model': pca,
        'scores': scores_df,
        'loadings': loadings_df,
        'explained_variance': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)
    }

    return results

def detect_pca_outliers(pca_results, conf_level=0.95):
    """
    Calculates Hotelling's T2 statistic for each sample to identify outliers
    in the PCA model space.

    Args:
        pca_results (dict): Output from perform_pca.
        conf_level (float): Confidence level (default 0.95).

    Returns:
        pd.DataFrame: A dataframe containing T2 values and a boolean 'is_outlier' flag.
    """
    scores = pca_results['scores']
    eigenvalues = pca_results['explained_variance']  # This is variance ratio, typically we need eigenvalues
    # Note: sklearn's explained_variance_ IS the eigenvalue
    model = pca_results['model']
    eigenvalues = model.explained_variance_

    # Calculate T2 for each sample
    # T2 = sum( (score_i^2) / eigenvalue_i ) for the selected components
    # We use all components calculated in the model for a robust check,
    # or just the first few (e.g., PC1, PC2). Usually calculated on the retained PCs.

    n_components = scores.shape[1]
    n_samples = scores.shape[0]

    # Formula: T^2 = Score^2 / Eigenvalue
    t2_values = np.sum((scores.values ** 2) / eigenvalues[:n_components], axis=1)

    # Calculate Critical Value (F-distribution based limit)
    # T2 limit = ((n-1)(n+1) / n(n-k)) * F_crit(k, n-k-1)
    F_crit = stats.f.ppf(conf_level, n_components, n_samples - n_components - 1)
    t2_limit = (n_components * (n_samples - 1) / (n_samples - n_components)) * F_crit

    outlier_df = pd.DataFrame({
        'T2': t2_values,
        'Limit': t2_limit,
        'is_outlier': t2_values > t2_limit
    }, index=scores.index)

    return outlier_df

def get_optimal_components(pca_results, variance_threshold=0.90):
    """
    Returns the number of components needed to explain the given variance threshold.
    """
    cum_var = pca_results['cumulative_variance']
    # np.argmax returns the index of the first True value
    n_components = np.argmax(cum_var >= variance_threshold) + 1
    return n_components

def plot_pca_scree(pca_results, output_dir, file_name="pca_scree_plot", threshold=0.9):
    """
    Generates a Scree Plot (Elbow Method) to help select the optimal number of components.

    Displays both individual Explained Variance (Bars) and Cumulative Variance (Line).
    """
    os.makedirs(output_dir, exist_ok=True)

    var_ratio = pca_results['explained_variance']
    cum_var = pca_results['cumulative_variance']
    n_pcs = len(var_ratio)

    # Limit to first 20 PCs for readability if many dimensions exist
    n_plot = min(20, n_pcs)
    x_range = np.arange(1, n_plot + 1)

    fig, ax1 = plt.subplots()

    # Bar Chart (Individual Variance)
    ax1.bar(x_range, var_ratio[:n_plot] * 100, color=DISCRETE_COLORS[0], alpha=0.7, label='Individual Variance')
    ax1.set_xlabel('Principal Components')
    ax1.set_ylabel('Explained Variance (%)', color=DISCRETE_COLORS[0])
    ax1.tick_params(axis='y', labelcolor=DISCRETE_COLORS[0])
    ax1.set_xticks(x_range)

    # Line Chart (Cumulative Variance)
    ax2 = ax1.twinx()
    ax2.plot(x_range, cum_var[:n_plot] * 100, color=DISCRETE_COLORS[4], marker='o', linewidth=2,
             label='Cumulative Variance')
    ax2.set_ylabel('Cumulative Variance (%)', color=DISCRETE_COLORS[4])
    ax2.tick_params(axis='y', labelcolor=DISCRETE_COLORS[4])

    # Threshold Line (e.g., 90%)
    ax2.axhline(y=threshold * 100, color='grey', linestyle='--', alpha=0.5)
    ax2.text(n_plot, threshold * 100 + 1, f'{int(threshold * 100)}% Threshold', color='grey', ha='right')

    #Legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')

    plt.title('PCA Scree Plot (Explained Variance)', fontweight='bold')

    save_path = os.path.join(output_dir, f"{file_name}.{SAVE_FORMAT}")
    plt.savefig(save_path, format=SAVE_FORMAT, bbox_inches='tight')
    plt.close(fig)
    print(f"Generating PCA Scree Plot saved to {save_path}")


def plot_pca_scores(pca_results, df, output_dir, pc_x=1, pc_y=2, file_name="pca_score_plot", class_col='Class',
                    show_ellipse=True):
    """
    Generates the PCA Score Plot with Hotelling's T2 Confidence Ellipse.

    THEORY:
    - Visualizes samples in the reduced space (PCx vs PCy).
    - Ellipse: Represents the 95% Confidence Interval for the Normal Distribution of the model.
      Strictly calculated on the GLOBAL dataset (unsupervised), not per-class.
    - Outliers: Samples outside the ellipse are potential candidates based on T2 distance.
    """
    os.makedirs(output_dir, exist_ok=True)

    scores = pca_results['scores']
    var_ratio = pca_results['explained_variance']

    # Get PC labels and data
    pc_x_label = f"PC{pc_x}"
    pc_y_label = f"PC{pc_y}"

    x_data = scores[pc_x_label]
    y_data = scores[pc_y_label]

    fig, ax = plt.subplots()

    #Handling of class NotShow if we don't want to see class in plotting
    mask_hidden = df[class_col] == 'NotShow'
    mask_visible = ~mask_hidden

    if mask_hidden.any():
        ax.scatter(x_data[mask_hidden], y_data[mask_hidden],
                   alpha=0, s=100, edgecolor='none')

    visible_classes = df.loc[mask_visible, class_col].unique()
    current_palette = DISCRETE_COLORS[:len(visible_classes)]
    markers_dict = {cls: MARKERS[i % len(MARKERS)] for i, cls in enumerate(visible_classes)}

    # Scatter Plot
    sns.scatterplot(x=x_data[mask_visible], y=y_data[mask_visible],
                    hue=df.loc[mask_visible, class_col],
                    style=df.loc[mask_visible, class_col],
                    palette=current_palette, markers=markers_dict,
                    s=100, alpha=0.8, edgecolor='k', ax=ax)

    # --- HOTELLING'S T2 ELLIPSE CALCULATION ---
    if show_ellipse:
        # The ellipse is calculated on the assumption of multivariate normality of the SCORES.
        # Since PCA scores are orthogonal (uncorrelated), the axes of the ellipse align with X and Y.
        # Width/Height are proportional to the square root of the eigenvalues (variance) * Chi-Square critical value.

        # Chi-Square critical value for 95% confidence with 2 Degrees of Freedom is ~5.991
        chi2_val = 5.991

        # Variance of the plotted PCs
        std_x = np.std(x_data)
        std_y = np.std(y_data)

        # Width and Height (Total length, so 2 * radius)
        width = 2 * np.sqrt(chi2_val) * std_x
        height = 2 * np.sqrt(chi2_val) * std_y

        ellipse = Ellipse((0, 0), width=width, height=height,
                          facecolor='none', edgecolor='black', linestyle='--', linewidth=1.5,
                          label='95% Confidence (Hotelling T²)')
        ax.add_patch(ellipse)

    # Labels with Explained Variance
    var_x = var_ratio[pc_x - 1] * 100
    var_y = var_ratio[pc_y - 1] * 100

    ax.set_xlabel(f"{pc_x_label} ({var_x:.1f}%)", fontweight='bold')
    ax.set_ylabel(f"{pc_y_label} ({var_y:.1f}%)", fontweight='bold')
    ax.set_title(f"PCA Score Plot: {pc_x_label} vs {pc_y_label}", fontweight='bold')

    # Center lines
    ax.axhline(0, color='grey', linestyle=':', linewidth=0.8)
    ax.axvline(0, color='grey', linestyle=':', linewidth=0.8)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{file_name}_PC{pc_x}vsPC{pc_y}.{SAVE_FORMAT}")
    plt.savefig(save_path, format=SAVE_FORMAT, bbox_inches='tight')
    plt.close(fig)
    print(f"Generating PCA Score Plot ({pc_x}vs{pc_y}) saved to {save_path}")


def plot_pca_loadings(pca_results, output_dir, pc_x=1, pc_y=2, file_name="pca_loading_plot", top_n=20):
    """
    Generates the PCA Loading Plot to interpret variable contributions.

    Shows the weight of each metabolite on the selected PCs.
    Since we have many features, we use a Scatter representation (Simpler Biplot concept).
    """
    os.makedirs(output_dir, exist_ok=True)

    loadings = pca_results['loadings']
    pc_x_label = f"PC{pc_x}"
    pc_y_label = f"PC{pc_y}"

    # Calculate magnitude (distance from origin) to identify top contributors
    magnitude = np.sqrt(loadings[pc_x_label] ** 2 + loadings[pc_y_label] ** 2)
    loadings['magnitude'] = magnitude

    # Sort and take top N for labeling
    top_loadings = loadings.sort_values(by='magnitude', ascending=False).head(top_n)

    fig, ax = plt.subplots()

    # Plot all features
    # Usiamo un colore secondario della palette (es. ind. 3) per lo sfondo, con trasparenza
    ax.scatter(loadings[pc_x_label], loadings[pc_y_label], color=DISCRETE_COLORS[3], alpha=0.7, s=20, label='All Features')

    # Plot top features as highlighted points
    if top_n > 0:
        ax.scatter(top_loadings[pc_x_label], top_loadings[pc_y_label], color=DISCRETE_COLORS[9], s=50,
                   label=f'Top {top_n} Contributors')

    # Annotate Top N
    texts = []
    for feature, row in top_loadings.iterrows():
        texts.append(plt.text(row[pc_x_label], row[pc_y_label], feature, fontsize=8, color=DISCRETE_COLORS[9],
                              fontweight='bold'))

    # Center lines
    ax.axhline(0, color='k', linestyle='-', linewidth=0.8)
    ax.axvline(0, color='k', linestyle='-', linewidth=0.8)

    # Draw circle of correlation (optional visual aid, radius depends on scaling, usually 1 or max loading)
    # Here we just keep axes symmetric
    max_val = max(abs(loadings[pc_x_label].max()), abs(loadings[pc_y_label].max())) * 1.1
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)

    ax.set_xlabel(f"Loading {pc_x_label}", fontweight='bold')
    ax.set_ylabel(f"Loading {pc_y_label}", fontweight='bold')
    ax.set_title(f"PCA Loading Plot: {pc_x_label} vs {pc_y_label}", fontweight='bold')
    ax.legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{file_name}_PC{pc_x}vsPC{pc_y}.{SAVE_FORMAT}")
    plt.savefig(save_path, format=SAVE_FORMAT, bbox_inches='tight')
    plt.close(fig)
    print(f"Generating PCA Loading Plot saved to {save_path}")


def plot_loading_profile(pca_results, output_dir, pc_index=1, file_name="loading_profile"):
    """
    Generates a Spectral-style Line Plot for the loadings of a SINGLE component.

    THEORY:
    - Visualizes the 'shape' or 'profile' of the component across all variables.
    - Particularly useful in Omics (LC-MS, NMR) to identify spectral regions or
      clusters of m/z values that drive the separation.
    - X-axis: Features (ordered by m/z or index).
    - Y-axis: Loading weight.
    """
    os.makedirs(output_dir, exist_ok=True)

    pc_label = f"PC{pc_index}"
    # Retrieve loadings for the specific PC
    loadings = pca_results['loadings'][pc_label]

    fig, ax = plt.subplots()

    # --- 1. Determine X-Axis (Numeric m/z or Sequential) ---
    try:
        # Try to convert feature names to floats (assuming they represent m/z or RT)
        x_values = loadings.index.astype(float)
        # Sort by X to ensure the line doesn't scribble back and forth
        sorted_indices = np.argsort(x_values)
        x_plot = x_values[sorted_indices]
        y_plot = loadings.values[sorted_indices]
        xlabel = "Feature (m/z or RT)"
    except ValueError:
        # If features are strings (e.g., "Met_1"), use sequential index
        x_plot = np.arange(len(loadings))
        y_plot = loadings.values
        xlabel = "Feature Index"

    # --- 2. Color Selection ---
    # Cycle through the palette based on PC index to give distinct colors
    # e.g., PC1 -> Color[0], PC2 -> Color[1], etc.
    line_color = DISCRETE_COLORS[(pc_index - 1) % len(DISCRETE_COLORS)]

    # --- 3. Plotting (Line Plot) ---
    ax.plot(x_plot, y_plot, color=line_color, linewidth=1.2, label=f'Loadings {pc_label}')

    # Add a horizontal line at 0 for reference
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

    # --- 4. Styling ---
    ax.set_title(f"Loadings Profile: {pc_label}", fontweight='bold')
    ax.set_ylabel(f"Loading Value")
    ax.set_xlabel(xlabel)

    # Add grid but keep it subtle
    ax.grid(True, linestyle=':', alpha=0.4)

    # Optional: Fill area under curve slightly for better visual impact
    # ax.fill_between(x_plot, 0, y_plot, color=line_color, alpha=0.1)

    # Legend
    ax.legend(loc='upper right')

    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{file_name}_{pc_label}.{SAVE_FORMAT}")
    plt.savefig(save_path, format=SAVE_FORMAT, bbox_inches='tight')
    plt.close(fig)
    print(f"Generating PCA Loading Profile ({pc_label}) saved to {save_path}")