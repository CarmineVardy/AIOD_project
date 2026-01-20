"""
Data Preprocessing Module.

This module provides a comprehensive suite of functions for normalizing, transforming,
and scaling metabolomics data.

Techniques implemented:
1. Sample Normalization: TIC, Median, Mean, Max, Range, PQN, Quantile.
2. Data Transformation: Log10, Log2, Natural Log (Ln), Square Root, Cube Root.
3. Data Scaling: Autoscaling (Unit Variance).
"""

import numpy as np
import pandas as pd


# =============================================================================
# NORMALIZATION FUNCTIONS
# =============================================================================

def normalization_tic(df):
    """
    Total Ion Count (TIC) Normalization.
    Divides each intensity by the sum of intensities in that sample.
    """
    numeric_df = df.drop(columns=['Class'])
    row_sums = numeric_df.sum(axis=1)
    mean_sum = row_sums.mean()

    norm_data = numeric_df.div(row_sums, axis=0) * mean_sum

    norm_df = norm_data.copy()
    norm_df.insert(0, 'Class', df['Class'])
    return norm_df


def normalization_median(df):
    """
    Median Normalization.
    Divides each intensity by the median intensity of that sample.
    Robust to outliers.
    """
    numeric_df = df.drop(columns=['Class'])
    row_medians = numeric_df.median(axis=1)
    mean_median = row_medians.mean()

    norm_data = numeric_df.div(row_medians, axis=0) * mean_median

    norm_df = norm_data.copy()
    norm_df.insert(0, 'Class', df['Class'])
    return norm_df


def normalization_mean(df):
    """
    Mean Normalization.
    Divides each intensity by the mean intensity of that sample.
    Sensitive to outliers.
    """
    numeric_df = df.drop(columns=['Class'])
    row_means = numeric_df.mean(axis=1)
    mean_mean = row_means.mean()

    norm_data = numeric_df.div(row_means, axis=0) * mean_mean

    norm_df = norm_data.copy()
    norm_df.insert(0, 'Class', df['Class'])
    return norm_df


def normalization_max(df):
    """
    Maximum Value Normalization.
    Divides each intensity by the maximum intensity of that sample.
    """
    numeric_df = df.drop(columns=['Class'])
    row_maxs = numeric_df.max(axis=1)
    mean_max = row_maxs.mean()

    norm_data = numeric_df.div(row_maxs, axis=0) * mean_max

    norm_df = norm_data.copy()
    norm_df.insert(0, 'Class', df['Class'])
    return norm_df


def normalization_range(df):
    """
    Range Normalization.
    Divides each intensity by the range (Max - Min) of that sample.
    """
    numeric_df = df.drop(columns=['Class'])
    row_ranges = numeric_df.max(axis=1) - numeric_df.min(axis=1)
    row_ranges = row_ranges.replace(0, 1)  # Safety check
    mean_range = row_ranges.mean()

    norm_data = numeric_df.div(row_ranges, axis=0) * mean_range

    norm_df = norm_data.copy()
    norm_df.insert(0, 'Class', df['Class'])
    return norm_df


def normalization_pqn(df):
    """
    Probabilistic Quotient Normalization (PQN).
    Methodology: Calculates a reference spectrum (median feature values), computes quotients,
    and normalizes by the median quotient (dilution factor).
    Gold standard for biofluids.
    """
    numeric_df = df.drop(columns=['Class'])

    # Reference Spectrum
    reference_spectrum = numeric_df.median(axis=0)
    reference_spectrum[reference_spectrum == 0] = 1.0

    # Quotients
    quotients = numeric_df.div(reference_spectrum, axis=1)

    # Dilution Factor
    dilution_factors = quotients.median(axis=1)

    norm_data = numeric_df.div(dilution_factors, axis=0)

    norm_df = norm_data.copy()
    norm_df.insert(0, 'Class', df['Class'])
    return norm_df


def normalization_quantile(df):
    """
    Quantile Normalization.
    Forces the distribution of all samples to be identical (based on rank).
    """
    numeric_df = df.drop(columns=['Class'])

    # Sort indices to allow reconstruction
    sorted_idxs = np.argsort(numeric_df.values, axis=1)

    # Sort data row-wise
    sorted_data = np.sort(numeric_df.values, axis=1)

    # Target distribution (mean of columns)
    mean_quantiles = np.mean(sorted_data, axis=0)

    # Map back
    norm_values = np.zeros_like(numeric_df.values)
    for i in range(len(norm_values)):
        norm_values[i, sorted_idxs[i]] = mean_quantiles

    norm_df = pd.DataFrame(norm_values, index=numeric_df.index, columns=numeric_df.columns)
    norm_df.insert(0, 'Class', df['Class'])
    return norm_df


# =============================================================================
# TRANSFORMATION FUNCTIONS
# =============================================================================

def transformation_log10(df):
    """
    Log10 Transformation: log10(x + 1).
    Strong compression of large values.
    """
    numeric_df = df.drop(columns=['Class'])
    transformed_data = np.log10(numeric_df + 1)

    df_trans = transformed_data.copy()
    df_trans.insert(0, 'Class', df['Class'])
    return df_trans


def transformation_log2(df):
    """
    Log2 Transformation: log2(x + 1).
    Interpretable as fold-change (1 unit = 2x).
    """
    numeric_df = df.drop(columns=['Class'])
    transformed_data = np.log2(numeric_df + 1)

    df_trans = transformed_data.copy()
    df_trans.insert(0, 'Class', df['Class'])
    return df_trans


def transformation_log_e(df):
    """
    Natural Logarithm (Ln) Transformation: ln(x + 1).
    Standard for statistical modeling.
    """
    numeric_df = df.drop(columns=['Class'])
    transformed_data = np.log(numeric_df + 1)

    df_trans = transformed_data.copy()
    df_trans.insert(0, 'Class', df['Class'])
    return df_trans


def transformation_sqrt(df):
    """
    Square Root Transformation: sqrt(x).
    Moderate compression, suitable for count data.
    """
    numeric_df = df.drop(columns=['Class'])
    transformed_data = np.sqrt(numeric_df)

    df_trans = transformed_data.copy()
    df_trans.insert(0, 'Class', df['Class'])
    return df_trans


def transformation_cuberoot(df):
    """
    Cube Root Transformation: x^(1/3).
    Handles negatives and heavy skewness.
    """
    numeric_df = df.drop(columns=['Class'])
    transformed_data = np.cbrt(numeric_df)

    df_trans = transformed_data.copy()
    df_trans.insert(0, 'Class', df['Class'])
    return df_trans


# =============================================================================
# SCALING FUNCTIONS
# =============================================================================

def scaling_autoscaling(df):
    """
    Autoscaling (Unit Variance Scaling): (x - mean) / std.
    Centers features at 0 with variance 1. Gives all metabolites equal weight.
    """
    numeric_df = df.drop(columns=['Class'])
    col_means = numeric_df.mean(axis=0)
    col_stds = numeric_df.std(axis=0)

    col_stds = col_stds.replace(0, 1)  # Safety check

    scaled_data = (numeric_df - col_means) / col_stds

    df_scaled = scaled_data.copy()
    df_scaled.insert(0, 'Class', df['Class'])
    return df_scaled