import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

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