import os

import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src.config_visualization import *


def perform_univariate_analysis(df, class_col='Class', test_type='ttest', equal_var=False, log2_transform_fc=True):
    """
    Performs univariate statistical analysis for each feature to identify significant differences
    between two groups (e.g., CTRL vs CHD).

    Theoretical Background:
    - Hypothesis Testing: H0 (means are equal) vs H1 (means differ).
    - Tests:
        - T-test (Parametric): Assumes normality. Welch's t-test (equal_var=False) is safer/default.
        - Mann-Whitney U (Non-Parametric): Rank-sum test, robust to non-normality and outliers.
    - Multiple Testing Correction: Essential in Omics (p >> n) to control False Discovery Rate (FDR).
      Uses Benjamini-Hochberg procedure.
    - Fold Change (FC): Measures magnitude of change. Log2(FC) allows symmetric visualization (Volcano Plot).

    Args:
        df (pd.DataFrame): Input dataframe containing sample features and the class column.
                           Must contain exactly 2 classes in the 'Class' column.
        class_col (str): Name of the column containing group labels.
        test_type (str): 'ttest' (Student/Welch) or 'mannwhitney' (Non-parametric).
        equal_var (bool): If True, assumes equal variance for T-test.
                          False (default) uses Welch's T-test, which is generally more robust.
        log2_transform_fc (bool): If True, returns Log2(Fold Change). If False, raw Fold Change.

    Returns:
        pd.DataFrame: A results dataframe indexed by feature name, containing:
                      - p_value
                      - p_value_adj (FDR corrected)
                      - fold_change (or log2_fold_change)
                      - mean_group1, mean_group2
                      - significant (Boolean flag based on adj p < 0.05)
    """

    # 1. Prepare Data Groups
    # Extract the two unique classes
    classes = df[class_col].unique()
    if len(classes) != 2:
        raise ValueError(f"Univariate analysis requires exactly 2 classes. Found {len(classes)}: {classes}")

    # Define groups based on sorted class names to ensure consistency (e.g., CHD vs CTRL)
    # Usually, we want Case vs Control. Sorting alphabetically might put CHD first, then CTRL.
    # Group 1 = First in sort (e.g., CHD), Group 2 = Second in sort (e.g., CTRL)
    # FC will be calculated as Group 1 / Group 2.
    group1_label, group2_label = sorted(classes)

    group1_data = df[df[class_col] == group1_label].drop(columns=[class_col])
    group2_data = df[df[class_col] == group2_label].drop(columns=[class_col])

    feature_names = group1_data.columns
    results = []

    # 2. Iterate over each feature to perform the test
    for feature in feature_names:
        g1_values = group1_data[feature]
        g2_values = group2_data[feature]

        # --- A. Statistical Test ---
        if test_type == 'ttest':
            # ttest_ind performs T-test for independent samples
            stat, p_val = stats.ttest_ind(g1_values, g2_values, equal_var=equal_var, nan_policy='omit')
        elif test_type == 'mannwhitney':
            # mannwhitneyu is the non-parametric alternative
            stat, p_val = stats.mannwhitneyu(g1_values, g2_values, alternative='two-sided', nan_policy='omit')
        else:
            raise ValueError("Invalid test_type. Choose 'ttest' or 'mannwhitney'.")

        # --- B. Fold Change Calculation ---
        # Calculate means for FC
        mean_g1 = g1_values.mean()
        mean_g2 = g2_values.mean()

        # Handle division by zero or negative values if data is not strictly positive
        # In metabolomics (peak areas), values are usually positive.
        # If mean_g2 is 0, FC is technically infinite. We handle this safely.
        if mean_g2 == 0:
            fc = np.nan  # Or a large number/infinity depending on preference
        else:
            fc = mean_g1 / mean_g2

        # Log2 Transformation for Volcano Plot readiness
        if log2_transform_fc:
            # Add a tiny epsilon if needed to avoid log(0), though means are rarely exactly 0 in raw MS data
            # Here we assume data is clean or we accept -inf for 0.
            if fc > 0:
                fc_val = np.log2(fc)
            else:
                fc_val = np.nan
            fc_col_name = 'log2_fc'
        else:
            fc_val = fc
            fc_col_name = 'fold_change'

        results.append({
            'Feature': feature,
            'p_value': p_val,
            f'mean_{group1_label}': mean_g1,
            f'mean_{group2_label}': mean_g2,
            fc_col_name: fc_val
        })

    # Create DataFrame from results
    results_df = pd.DataFrame(results).set_index('Feature')

    # 3. Multiple Testing Correction (FDR)
    # We drop NaNs before correction to avoid errors, then join back if necessary,
    # but typically we just operate on valid p-values.
    mask_valid = results_df['p_value'].notna()
    p_values_valid = results_df.loc[mask_valid, 'p_value']

    # multipletests returns: reject (bool array), pvals_corrected, ...
    reject, pvals_corrected, _, _ = multipletests(p_values_valid, alpha=0.05, method='fdr_bh')

    # Assign back to DataFrame
    results_df.loc[mask_valid, 'p_value_adj'] = pvals_corrected
    results_df['significant'] = results_df['p_value_adj'] < 0.05

    # Sort by p-value for readability
    results_df = results_df.sort_values(by='p_value')

    return results_df

def plot_volcano(results_df, output_dir, file_name="volcano_plot",
                 p_thresh=0.05, fc_thresh=2.0, use_adj_pval=True, top_n_labels=10):
    """
    Generates a Volcano Plot to visualize Univariate Analysis results.

    THEORY & IMPLEMENTATION DETAILS:
    --------------------------------
    1. Axes Definition:
       - X-axis: Magnitude of change (Log2 Fold Change).
         Positive = Up-regulated in Group 1 vs Group 2.
         Negative = Down-regulated.
       - Y-axis: Statistical Significance (-Log10 p-value).
         Higher values = More significant.
    2. Thresholds (Regions of Interest):
       - Horizontal Line: Statistical significance cutoff (e.g., p < 0.05).
         Can use raw p-value or FDR-adjusted p-value (recommended).
       - Vertical Lines: Biological relevance cutoff (Fold Change).
         Input is linear FC (e.g., 2.0), converted to +/- Log2(FC) for plotting.
    3. Quadrants:
       - Top-Right: Significantly Up-regulated.
       - Top-Left: Significantly Down-regulated.
       - Bottom / Center: Not significant (Noise).

    Args:
        results_df (pd.DataFrame): Output from 'perform_univariate_analysis'.
                                   Must contain 'p_value' (or 'p_value_adj') and 'log2_fc'.
        output_dir (str): Output directory.
        file_name (str): Filename.
        p_thresh (float): Significance threshold (default 0.05).
        fc_thresh (float): Fold Change threshold (linear, e.g., 2.0 implies 2x change).
        use_adj_pval (bool): If True, uses 'p_value_adj' (FDR). If False, uses raw 'p_value'.
        top_n_labels (int): Number of top significant features to annotate with text.
    """

    os.makedirs(output_dir, exist_ok=True)

    # 1. Prepare Data for Plotting
    df = results_df.copy()

    # Select which p-value to use
    pval_col = 'p_value_adj' if use_adj_pval else 'p_value'

    # Calculate -Log10(p-value) for Y-axis
    # Adding a tiny epsilon to avoid log(0) if p-value is extremely small
    df['neg_log10_p'] = -np.log10(df[pval_col] + 1e-300)

    # Calculate Log2 threshold from linear FC input
    # e.g., FC=2.0 -> log2(2)=1.0. We check for > 1.0 and < -1.0
    log2_fc_thresh = np.log2(fc_thresh)
    neg_log10_p_thresh = -np.log10(p_thresh)

    # 2. Categorize Points (Up, Down, NS)
    # Define conditions
    cond_sig = df[pval_col] < p_thresh
    cond_up = df['log2_fc'] > log2_fc_thresh
    cond_down = df['log2_fc'] < -log2_fc_thresh

    conditions = [
        (cond_sig & cond_up),
        (cond_sig & cond_down)
    ]
    choices = ['UP', 'DOWN']
    df['status'] = np.select(conditions, choices, default='NS')

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(10, 8))

    # Map colors: NS -> Grey, UP -> Red/Yellow-ish, DOWN -> Blue/Purple-ish
    # We pick colors from our global DISCRETE_COLORS or define specific ones for contrast
    # Using 'tab10' or manual mapping for clear distinction
    palette = {
        'NS': 'lightgrey',
        'UP': '#d62728',  # Red
        'DOWN': '#1f77b4'  # Blue
    }

    sns.scatterplot(data=df, x='log2_fc', y='neg_log10_p',
                    hue='status', style='status',
                    palette=palette,
                    hue_order=['NS', 'UP', 'DOWN'],
                    markers={'NS': 'o', 'UP': '^', 'DOWN': 'v'},
                    s=60, alpha=0.8, edgecolor='k', linewidth=0.5,
                    ax=ax)

    # 4. Add Threshold Lines
    # Horizontal (Significance)
    ax.axhline(neg_log10_p_thresh, color='k', linestyle='--', linewidth=1, alpha=0.7)
    # Vertical (Fold Change)
    ax.axvline(log2_fc_thresh, color='k', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(-log2_fc_thresh, color='k', linestyle='--', linewidth=1, alpha=0.7)

    # 5. Annotate Top Features
    # We sort by p-value (ascending) to find the most significant ones
    top_hits = df[df['status'] != 'NS'].sort_values(by=pval_col).head(top_n_labels)

    texts = []
    for idx, row in top_hits.iterrows():
        texts.append(plt.text(row['log2_fc'], row['neg_log10_p'], idx, fontsize=9))

    # Adjust text positions to avoid overlap (requires 'adjustText' library if available,
    # otherwise basic placement. Here we assume standard matplotlib).
    # If you have adjustText installed: from adjustText import adjust_text; adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))

    # 6. Labels and Title
    pval_label = "FDR (adj. p-value)" if use_adj_pval else "Raw p-value"
    ax.set_xlabel(r"$Log_2$ Fold Change", fontweight='bold')
    ax.set_ylabel(r"$-Log_{10}$ " + f"({pval_label})", fontweight='bold')

    title_str = f"Volcano Plot (Thresh: p<{p_thresh}, FC>{fc_thresh})"
    ax.set_title(title_str, fontweight='bold')

    # Add informative legend text about quadrants
    info_text = (f"UP: p<{p_thresh} & FC>{fc_thresh}\n"
                 f"DOWN: p<{p_thresh} & FC<1/{fc_thresh}")
    plt.text(0.98, 0.02, info_text, transform=ax.transAxes,
             fontsize=9, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.9))

    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{file_name}.{SAVE_FORMAT}")
    plt.savefig(save_path, format=SAVE_FORMAT, bbox_inches='tight')
    plt.close(fig)

    print(f"Generating Volcano Plot saved to {save_path}")
    print(f"   Significant features highlighted: {len(df[df['status'] != 'NS'])}")