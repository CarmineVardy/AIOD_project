import os
import pandas as pd
import numpy as np
from scipy.stats import iqr, sem
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats

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

    # Separation of Target and Features
    # We assume 'Class' is the target column.
    if 'Class' in df.columns:
        classes = df['Class']
        # Select only numeric columns for data integrity checks
        numeric_data = df.drop(columns=['Class'])
    else:
        # Fallback if Class column is missing (though it shouldn't be based on project structure)
        classes = pd.Series(["Unknown"] * len(df), index=df.index)
        numeric_data = df.select_dtypes(include=[np.number])

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