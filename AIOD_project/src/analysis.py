"""
Analysis Module.

This module provides functions for:
1. Dataset Characterization: Summary reports on dimensions, class balance, and missing values.
2. Feature Statistics: Calculation of descriptive stats (Mean, Median, CV, Skewness, Kurtosis).
3. Univariate Analysis: Hypothesis testing (T-test/Mann-Whitney) and Fold Change calculation with FDR correction.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import sem
from statsmodels.stats.multitest import multipletests


def generate_dataset_report(df, dataset_name):
    """
    Analyzes the dataset structure and returns a summary report.

    Args:
        df (pd.DataFrame): Input dataframe. Index must contain sample names,
                           and a 'Class' column must be present.
        dataset_name (str): Name of the dataset to label the report.

    Returns:
        pd.DataFrame: A dataframe with columns ['Metric', 'Value'] containing summary stats.
    """
    classes = df['Class']
    numeric_data = df.drop(columns=['Class'])
    sample_names = df.index
    feature_names = numeric_data.columns

    report_data = []

    # 1. Overview
    report_data.append({'Metric': 'Dataset Name', 'Value': dataset_name})
    report_data.append({'Metric': 'Total Features (Metabolites)', 'Value': len(feature_names)})
    report_data.append({'Metric': 'Total Samples', 'Value': len(sample_names)})

    # 2. Class Distribution
    for cls, count in classes.value_counts().items():
        report_data.append({'Metric': f'Class Count: {cls}', 'Value': count})

    # 3. Nomenclature Analysis (Biological vs Technical Replicates)
    samples_str = sample_names.astype(str)

    # Initialize counters
    ctrl_primary = 0
    ctrl_replicate = 0
    chd_primary = 0
    chd_replicate = 0
    qc_count = 0

    with_suffix_00 = 0
    with_suffix_01 = 0
    without_suffix = 0

    for name, label in zip(samples_str, classes):
        # Suffix categorization
        if '_00' in name:
            with_suffix_00 += 1
        elif '_01' in name:
            with_suffix_01 += 1
        else:
            without_suffix += 1

        # Class-specific replicate logic
        if label == 'QC':
            qc_count += 1
            continue

        is_replicate = '_01' in name

        if label == 'CTRL':
            if is_replicate:
                ctrl_replicate += 1
            else:
                ctrl_primary += 1
        elif label == 'CHD':
            if is_replicate:
                chd_replicate += 1
            else:
                chd_primary += 1

    total_biological = with_suffix_00 + without_suffix

    report_data.append({'Metric': "Samples with suffix '_00'", 'Value': with_suffix_00})
    report_data.append({'Metric': "Samples with suffix '_01' (Tech Rep)", 'Value': with_suffix_01})
    report_data.append({'Metric': "Samples without standard suffix", 'Value': without_suffix})
    report_data.append({'Metric': "Est. Unique Biological Samples", 'Value': total_biological})

    report_data.append({'Metric': 'CTRL - Biological Samples', 'Value': ctrl_primary})
    report_data.append({'Metric': 'CTRL - Technical Replicates', 'Value': ctrl_replicate})
    report_data.append({'Metric': 'CHD - Biological Samples', 'Value': chd_primary})
    report_data.append({'Metric': 'CHD - Technical Replicates', 'Value': chd_replicate})
    report_data.append({'Metric': 'QC - Total Samples', 'Value': qc_count})

    # 4. Data Integrity
    total_cells = numeric_data.size
    missing_values = numeric_data.isnull().sum().sum()
    missing_percentage = (missing_values / total_cells * 100) if total_cells > 0 else 0
    has_negative = (numeric_data < 0).any().any()

    report_data.append({'Metric': 'Missing Values (Count)', 'Value': missing_values})
    report_data.append({'Metric': 'Missing Values (%)', 'Value': f"{missing_percentage:.2f}%"})
    report_data.append({'Metric': 'Negative Values Present', 'Value': "YES" if has_negative else "NO"})

    return pd.DataFrame(report_data)


def compute_feature_statistics(df):
    """
    Calculates comprehensive descriptive statistics for each feature (metabolite).
    Includes measures of central tendency, variability (CV, MAD, IQR), and shape.

    Args:
        df (pd.DataFrame): Input dataframe (Samples x Features).

    Returns:
        pd.DataFrame: A dataframe where index corresponds to feature names and columns
                      to the calculated statistics.
    """
    df_numeric = df.drop(columns=['Class'])

    # Initialize with basic stats
    stats_df = df_numeric.describe().T

    # Variability Measures
    stats_df['variance'] = df_numeric.var()
    # Mean Absolute Deviation (MAD): robust alternative to std
    stats_df['mad'] = df_numeric.apply(lambda x: (x - x.mean()).abs().mean())
    stats_df['range'] = stats_df['max'] - stats_df['min']
    stats_df['iqr'] = stats_df['75%'] - stats_df['25%']

    # Coefficient of Variation (CV %): critical for QC
    stats_df['cv_percent'] = (stats_df['std'] / stats_df['mean'].replace(0, np.nan)) * 100

    return stats_df


def compute_correlation_matrix(df, method='pearson'):
    """
    Computes the correlation matrix to analyze relationships between features.

    Args:
        df (pd.DataFrame): Input dataframe.
        method (str): 'pearson' (linear), 'spearman', or 'kendall'.

    Returns:
        pd.DataFrame: Symmetric correlation matrix.
    """
    return df.drop(columns=['Class']).corr(method=method)


def identify_low_variance_features(stats_df, threshold_cv=30.0):
    """
    Identifies features with high variability (CV > threshold) which might indicate instability.

    Args:
        stats_df (pd.DataFrame): Output from compute_feature_statistics.
        threshold_cv (float): Cutoff for Coefficient of Variation (%).

    Returns:
        list: Names of unstable features.
    """
    return stats_df[stats_df['cv_percent'] > threshold_cv].index.tolist()


def perform_univariate_analysis(df, class_col='Class', test_type='ttest', equal_var=False, log2_transform_fc=True):
    """
    Performs univariate statistical analysis to compare two groups.
    Includes Hypothesis Testing, Fold Change calculation, and FDR correction.

    Args:
        df (pd.DataFrame): Input dataframe. Must contain exactly 2 classes.
        class_col (str): Column name for group labels.
        test_type (str): 'ttest' (Welch's/Student's) or 'mannwhitney'.
        equal_var (bool): If False, performs Welch's t-test (robust to unequal variances).
        log2_transform_fc (bool): If True, returns Log2(Fold Change).

    Returns:
        pd.DataFrame: Results indexed by feature (p-values, FDR adjusted p-values, Fold Change).
    """
    classes = df[class_col].unique()
    if len(classes) != 2:
        raise ValueError(f"Univariate analysis requires exactly 2 classes. Found: {classes}")

    # Sort to define Group 1 and Group 2 consistently (e.g., CHD vs CTRL)
    group1_label, group2_label = sorted(classes)

    group1_data = df[df[class_col] == group1_label].drop(columns=[class_col])
    group2_data = df[df[class_col] == group2_label].drop(columns=[class_col])

    feature_names = group1_data.columns
    results = []

    for feature in feature_names:
        g1_values = group1_data[feature]
        g2_values = group2_data[feature]

        # 1. Statistical Test
        if test_type == 'ttest':
            stat, p_val = stats.ttest_ind(g1_values, g2_values, equal_var=equal_var, nan_policy='omit')
        elif test_type == 'mannwhitney':
            stat, p_val = stats.mannwhitneyu(g1_values, g2_values, alternative='two-sided', nan_policy='omit')
        else:
            raise ValueError("Invalid test_type. Choose 'ttest' or 'mannwhitney'.")

        # 2. Fold Change Calculation
        mean_g1 = g1_values.mean()
        mean_g2 = g2_values.mean()

        if mean_g2 == 0:
            fc = np.nan
        else:
            fc = mean_g1 / mean_g2

        if log2_transform_fc:
            fc_val = np.log2(fc) if fc > 0 else np.nan
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

    results_df = pd.DataFrame(results).set_index('Feature')

    # 3. Multiple Testing Correction (Benjamini-Hochberg FDR)
    mask_valid = results_df['p_value'].notna()
    p_values_valid = results_df.loc[mask_valid, 'p_value']

    reject, pvals_corrected, _, _ = multipletests(p_values_valid, alpha=0.05, method='fdr_bh')

    results_df.loc[mask_valid, 'p_value_adj'] = pvals_corrected
    results_df['significant'] = results_df['p_value_adj'] < 0.05

    return results_df.sort_values(by='p_value')