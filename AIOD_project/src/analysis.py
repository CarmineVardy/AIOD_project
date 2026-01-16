import numpy as np
import pandas as pd
from scipy.stats import sem


def generate_dataset_report(df, dataset_name):
    """
    Analyzes the dataset structure and returns a summary report as a DataFrame.

    The report includes:
    - General overview (dimensions).
    - Class distribution.
    - Nomenclature analysis (Biological vs Technical replicates based on suffixes).
    - Data integrity checks (Missing values, Negative values).

    Args:
        df (pd.DataFrame): Input dataframe. Index must contain sample names,
                           and a 'Class' column must be present.
        dataset_name (str): Name of the dataset to label the report.

    Returns:
        pd.DataFrame: A dataframe with columns ['Metric', 'Value'] containing the summary stats.
    """

    classes = df['Class']
    numeric_data = df.drop(columns=['Class'])

    sample_names = df.index
    feature_names = numeric_data.columns

    # --- LIST TO COLLECT METRICS ---
    report_data = []

    # 1. OVERVIEW
    n_features = len(feature_names)
    n_samples_total = len(sample_names)

    report_data.append({'Metric': 'Dataset Name', 'Value': dataset_name})
    report_data.append({'Metric': 'Total Features (Metabolites)', 'Value': n_features})
    report_data.append({'Metric': 'Total Samples', 'Value': n_samples_total})

    # 2. CLASS DISTRIBUTION (Raw Counts)
    class_counts = classes.value_counts()
    for cls, count in class_counts.items():
        report_data.append({'Metric': f'Class Count: {cls}', 'Value': count})

    # 3. NOMENCLATURE ANALYSIS & 4. DETAILED BREAKDOWN
    # Logic to distinguish between Primary samples (e.g., '_00') and Technical Replicates ('_01')
    samples_str = sample_names.astype(str)

    # Counters
    ctrl_primary = 0
    ctrl_replicate = 0
    chd_primary = 0
    chd_replicate = 0
    qc_count = 0

    # General suffix counters
    with_suffix_00 = 0
    with_suffix_01 = 0
    without_suffix = 0

    for name, label in zip(samples_str, classes):
        # General Suffix Stats
        if '_00' in name:
            with_suffix_00 += 1
        elif '_01' in name:
            with_suffix_01 += 1
        else:
            without_suffix += 1

        # Detailed Class + Replicate Logic
        if label == 'QC':
            qc_count += 1
            continue  # QC usually doesn't have the _00/_01 biological distinction in this context

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

    # Appending Nomenclature Stats
    report_data.append({'Metric': "Samples with suffix '_00'", 'Value': with_suffix_00})
    report_data.append({'Metric': "Samples with suffix '_01' (Tech Rep)", 'Value': with_suffix_01})
    report_data.append({'Metric': "Samples without standard suffix", 'Value': without_suffix})
    report_data.append({'Metric': "Est. Unique Biological Samples", 'Value': total_biological})

    # Appending Detailed Breakdown
    report_data.append({'Metric': 'CTRL - Biological Samples', 'Value': ctrl_primary})
    report_data.append({'Metric': 'CTRL - Technical Replicates', 'Value': ctrl_replicate})
    report_data.append({'Metric': 'CHD - Biological Samples', 'Value': chd_primary})
    report_data.append({'Metric': 'CHD - Technical Replicates', 'Value': chd_replicate})
    report_data.append({'Metric': 'QC - Total Samples', 'Value': qc_count})

    # 5. DATA INTEGRITY
    total_cells = n_features * n_samples_total
    # Check for missing values (NaN)
    missing_values = numeric_data.isnull().sum().sum()
    missing_percentage = (missing_values / total_cells) * 100 if total_cells > 0 else 0
    # Check for negative values (impossible for peak areas/intensities)
    has_negative = (numeric_data < 0).any().any()

    report_data.append({'Metric': 'Missing Values (Count)', 'Value': missing_values})
    report_data.append({'Metric': 'Missing Values (%)', 'Value': f"{missing_percentage:.2f}%"})
    report_data.append({'Metric': 'Negative Values Present', 'Value': "YES" if has_negative else "NO"})

    # --- RETURN DATAFRAME ---
    return pd.DataFrame(report_data)

def compute_feature_statistics(df):
    """
    Calculates comprehensive descriptive statistics for each feature (metabolite) in the dataset.

    Theoretical Background:
    - Central Tendency: Mean (sensitive to outliers), Median (robust).
    - Variability (Central): Std, Variance, MAD (Mean Absolute Deviation).
    - Variability (Interval): Range (Max-Min), IQR (Interquartile Range - robust).
    - Relative Variability: CV (Coefficient of Variation) - crucial for Omics QC.
    - Shape: Skewness (asymmetry), Kurtosis (tailedness).

    Args:
        df (pd.DataFrame): Input dataframe where rows are samples and columns are features.
                           Data should be numerical (intensities or concentrations).

    Returns:
        pd.DataFrame: A dataframe where index corresponds to feature names and columns
                      to the calculated statistics.
    """
    # Separate numeric data (local reassignment, original df is safe)
    df = df.drop(columns=['Class'])

    # 1. Initialize the statistics DataFrame
    # We use the transpose of describe() as a starting point (gives count, mean, std, min, 25%, 50%, 75%, max)
    stats = df.describe().T

    # 2. Add Variability Measures

    # Variance (Square of Std)
    stats['variance'] = df.var()

    # Mean Absolute Deviation (MAD)
    # Measure of the average absolute distance between each data point and the mean.
    # More robust to outliers than variance because it doesn't square the differences.
    # Note: pandas .mad() is deprecated, so we implement it manually: mean(|x - mean|)
    stats['mad'] = df.apply(lambda x: (x - x.mean()).abs().mean())

    # Range (Max - Min)
    stats['range'] = stats['max'] - stats['min']

    # Interquartile Range (IQR)
    # Distance between 75th percentile (Q3) and 25th percentile (Q1).
    # Robust measure of variability.
    stats['iqr'] = stats['75%'] - stats['25%']

    # Coefficient of Variation (CV %)
    # Ratio of standard deviation to the mean. Standardized measure of dispersion.
    # Extremely important in metabolomics to assess feature stability (QC).
    # We handle division by zero or very small means.
    stats['cv_percent'] = (stats['std'] / stats['mean'].replace(0, np.nan)) * 100

    # 3. Add Shape Measures

    # Skewness
    # Measure of asymmetry. 0 = symmetric, >0 = right tail, <0 = left tail.
    stats['skewness'] = df.skew()

    # Kurtosis
    # Measure of "tailedness". High kurtosis = heavy tails/outliers.
    stats['kurtosis'] = df.kurt()

    # 4. Standard Error of the Mean (SEM)
    # Measures how far the sample mean of the data is likely to be from the true population mean.
    stats['sem'] = df.apply(sem)

    return stats

def compute_correlation_matrix(df, method='pearson'):
    """
    Computes the correlation matrix to analyze Mutual Variability between features.

    Theoretical Background:
    - Covariance measures direction of linear relationship but is scale-dependent.
    - Correlation (Pearson) is standardized Covariance (-1 to +1).

    Args:
        df (pd.DataFrame): Input dataframe.
        method (str): 'pearson' (linear), 'spearman' (rank-based, non-linear), or 'kendall'.

    Returns:
        pd.DataFrame: Square symmetric matrix of correlation coefficients.
    """
    # Separate numeric data
    df = df.drop(columns=['Class'])
    # Returns the correlation matrix (p x p features)
    return df.corr(method=method)

def identify_low_variance_features(stats_df, threshold_cv=30.0):
    """
    Helper function to identify features that might need to be removed based on CV.

    Args:
        stats_df (pd.DataFrame): Output from compute_feature_statistics.
        threshold_cv (float): Cutoff for Coefficient of Variation in percentage.

    Returns:
        list: Names of features with CV higher than the threshold (unstable)
              or extremely low variance.
    """
    # Filter features where CV > threshold
    unstable_features = stats_df[stats_df['cv_percent'] > threshold_cv].index.tolist()
    return unstable_features
