import pandas as pd
import numpy as np

def normalization_tic(df):
    """
    Total Ion Count (TIC) / Total Area Normalization.

    Method:
    1. Calculates the sum of all peak intensities for each sample (row).
    2. Divides each metabolite intensity by the total sum of that sample.
    3. Multiplies by the mean of all sums to maintain readable scale.
    """
    # Separate numeric data
    numeric_df = df.drop(columns=['Class'])

    # Calculate Sum per row
    row_sums = numeric_df.sum(axis=1)
    mean_sum = row_sums.mean()

    # Normalize
    norm_data = numeric_df.div(row_sums, axis=0) * mean_sum

    # Reattach Class
    norm_df = norm_data.copy()
    norm_df.insert(0, 'Class', df['Class'])
    return norm_df


def normalization_median(df):
    """
    Median Normalization.

    Method:
    1. Calculates the median intensity for each sample (row).
    2. Divides each metabolite by the sample's median.

    Pros: More robust to outliers than Mean or TIC.
    """
    numeric_df = df.drop(columns=['Class'])

    # Calculate Median per row
    row_medians = numeric_df.median(axis=1)
    mean_median = row_medians.mean()

    # Normalize
    norm_data = numeric_df.div(row_medians, axis=0) * mean_median

    norm_df = norm_data.copy()
    norm_df.insert(0, 'Class', df['Class'])
    return norm_df


def normalization_mean(df):
    """
    Mean Normalization.

    Method:
    1. Calculates the mean intensity for each sample.
    2. Divides each metabolite by the sample's mean.

    Cons: Very sensitive to outliers (high peaks).
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

    Method:
    1. Identifies the maximum intensity value in each sample.
    2. Divides all features in that sample by the maximum value.

    Cons: Extremely sensitive to outliers.
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

    Method:
    1. Calculates the range (Max - Min) for each sample.
    2. Divides features by this range.
    """
    numeric_df = df.drop(columns=['Class'])

    # Calculate Range (Max - Min)
    row_ranges = numeric_df.max(axis=1) - numeric_df.min(axis=1)

    # Handle division by zero if range is 0 (unlikely in real data but safer)
    row_ranges = row_ranges.replace(0, 1)

    mean_range = row_ranges.mean()

    norm_data = numeric_df.div(row_ranges, axis=0) * mean_range

    norm_df = norm_data.copy()
    norm_df.insert(0, 'Class', df['Class'])
    return norm_df


def normalization_pqn(df):
    """
    Probabilistic Quotient Normalization (PQN).

    Method:
    1. Calculates a Reference Spectrum (median of each feature across all samples).
    2. Calculates quotients (Sample_Intensity / Reference_Intensity).
    3. The dilution factor is the Median of these quotients for each sample.
    4. Divides the sample by its dilution factor.

    Pros: Gold standard for urine/plasma; robust to dilution and outliers.
    """
    numeric_df = df.drop(columns=['Class'])

    # 1. Reference Spectrum (Column-wise median)
    reference_spectrum = numeric_df.median(axis=0)

    # Avoid division by zero if a feature is always 0
    reference_spectrum[reference_spectrum == 0] = 1.0

    # 2. Quotients
    quotients = numeric_df.div(reference_spectrum, axis=1)

    # 3. Dilution Factor (Row-wise median of quotients)
    dilution_factors = quotients.median(axis=1)

    # 4. Normalize
    norm_data = numeric_df.div(dilution_factors, axis=0)

    norm_df = norm_data.copy()
    norm_df.insert(0, 'Class', df['Class'])
    return norm_df


def normalization_quantile(df):
    """
    Quantile Normalization.

    Method:
    Forces the distribution of lengths of every sample to be the same.
    1. Sorts each sample.
    2. Takes the mean across samples at each rank.
    3. Replaces original values with the mean of that rank.

    Pros: Powerful for array data.
    Cons: Can destroy biological signal if distribution shapes vary naturally.
    """
    numeric_df = df.drop(columns=['Class'])

    # 1. Get the rank of each value in its row (to reconstruct order later)
    # We use argsort to get indices that would sort the array
    sorted_idxs = np.argsort(numeric_df.values, axis=1)

    # 2. Sort the data (Row-wise)
    sorted_data = np.sort(numeric_df.values, axis=1)

    # 3. Calculate the mean of each column (Rank Mean)
    # This creates the "target distribution"
    mean_quantiles = np.mean(sorted_data, axis=0)

    # 4. Map back to original positions
    # Create an empty array of the same shape
    norm_values = np.zeros_like(numeric_df.values)

    # For each row, assign the mean_quantile values based on the original rank
    for i in range(len(norm_values)):
        # The logic here: put the target values into the positions determined by argsort
        norm_values[i, sorted_idxs[i]] = mean_quantiles

    norm_df = pd.DataFrame(norm_values, index=numeric_df.index, columns=numeric_df.columns)
    norm_df.insert(0, 'Class', df['Class'])

    return norm_df


def transformation_log10(df):
    """
    Log10 Transformation.

    Formula:
        x_new = log10(x + 1)

    Rationale:
    1. Converts right-skewed distributions (typical in metabolomics) into
       Gaussian-like distributions.
    2. Reduces the impact of large outliers (drastic compression).
    3. The '+ 1' offset handles zero values.
    """
    numeric_df = df.drop(columns=['Class'])
    transformed_data = np.log10(numeric_df + 1)

    df_trans = transformed_data.copy()
    df_trans.insert(0, 'Class', df['Class'])
    return df_trans


def transformation_log2(df):
    """
    Log2 Transformation.

    Formula:
        x_new = log2(x + 1)

    Rationale:
    1. Similar benefits to Log10 but scales data differently.
    2. Often preferred in biology because a difference of 1 unit
       corresponds to a 2-fold change in concentration.
    """
    numeric_df = df.drop(columns=['Class'])
    transformed_data = np.log2(numeric_df + 1)

    df_trans = transformed_data.copy()
    df_trans.insert(0, 'Class', df['Class'])
    return df_trans


def transformation_log_e(df):
    """
    Natural Logarithm (Ln) Transformation.

    Formula:
        x_new = ln(x + 1)  (where ln is log base e)

    Rationale:
    1. The natural logarithm uses Euler's number (e ≈ 2.718) as the base.
    2. Extremely common in statistical modeling and hypothesis testing (t-tests, ANOVA)
       because many statistical distributions converge to Normal via Ln.
    3. The '+ 1' offset handles zero values.
    """
    numeric_df = df.drop(columns=['Class'])

    # np.log in Python calcola il logaritmo naturale (base e)
    transformed_data = np.log(numeric_df + 1)

    df_trans = transformed_data.copy()
    df_trans.insert(0, 'Class', df['Class'])
    return df_trans

def transformation_sqrt(df):
    """
    Square Root Transformation.

    Formula:
        x_new = sqrt(x)

    Rationale:
    1. Compresses large values less aggressively than Log transformation.
    2. Often used for data following a Poisson distribution (counts).
    3. Handles zeros natively without needing an offset (sqrt(0) = 0).
    """
    numeric_df = df.drop(columns=['Class'])
    transformed_data = np.sqrt(numeric_df)

    df_trans = transformed_data.copy()
    df_trans.insert(0, 'Class', df['Class'])
    return df_trans


def transformation_cuberoot(df):
    """
    Cube Root Transformation.

    Formula:
        x_new = x^(1/3)

    Rationale:
    1. Useful for extremely skewed data but less aggressive than Log.
    2. Can handle negative values (though rare in raw peak areas) without creating NaNs.
    """
    numeric_df = df.drop(columns=['Class'])
    # Using np.cbrt is safer and cleaner than **(1/3) for handling negatives if any
    transformed_data = np.cbrt(numeric_df)

    df_trans = transformed_data.copy()
    df_trans.insert(0, 'Class', df['Class'])
    return df_trans


def scaling_autoscaling(df):
    """
    Autoscaling (Unit Variance Scaling).

    Formula:
        x_new = (x - mean) / std_dev

    Method:
    1. Mean Centering: Subtracts the column mean.
    2. Scaling: Divides by the column standard deviation.

    Rationale:
    - Gives every metabolite equal importance (weight = 1).
    - Crucial when features have vastly different ranges.
    - Risk: Can amplify noise (features with low signal but high noise become important).
    """
    numeric_df = df.drop(columns=['Class'])
    col_means = numeric_df.mean(axis=0)
    col_stds = numeric_df.std(axis=0)

    # Avoid division by zero
    col_stds = col_stds.replace(0, 1)

    scaled_data = (numeric_df - col_means) / col_stds

    df_scaled = scaled_data.copy()
    df_scaled.insert(0, 'Class', df['Class'])
    return df_scaled

def scaling_centering(df):
    """
    Mean Centering (Only).

    Formula:
        x_new = x - mean

    Method:
    1. Calculates the mean of each feature (column).
    2. Subtracts the mean from each value.

    Rationale:
    - Moves the center of the coordinate system to the center of the data.
    - Focuses the analysis on the *fluctuation* around the mean rather than the absolute value.
    - Does NOT correct for magnitude differences (large peaks still dominate).
    """
    numeric_df = df.drop(columns=['Class'])
    col_means = numeric_df.mean(axis=0)

    scaled_data = numeric_df - col_means

    df_scaled = scaled_data.copy()
    df_scaled.insert(0, 'Class', df['Class'])
    return df_scaled


def scaling_range(df):
    """
    Range Scaling (Min-Max Scaling).

    Formula:
        x_new = (x - min) / (max - min)

    Method:
    1. Identifies min and max for each feature.
    2. Scales values so they all fall within the range [0, 1] (or [-1, 1] if centered).

    Rationale:
    - Useful when comparing metabolites to biological limits.
    - Sensitive to outliers (a single outlier can compress all other data points).
    """
    numeric_df = df.drop(columns=['Class'])
    col_mins = numeric_df.min(axis=0)
    col_maxs = numeric_df.max(axis=0)

    denom = col_maxs - col_mins
    denom = denom.replace(0, 1)

    scaled_data = (numeric_df - col_mins) / denom

    df_scaled = scaled_data.copy()
    df_scaled.insert(0, 'Class', df['Class'])
    return df_scaled

