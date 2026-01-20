"""
Data Loading and Cleaning Module.

This module handles the ingestion of raw metabolomics data (CSV),
structural reshaping (transposition), and preliminary cleaning operations
such as the removal of Quality Control (QC) samples and technical replicates.
"""

import os
import pandas as pd


def load_raw_dataset(file_path):
    """
    Loads the raw CSV file from the specified path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: Raw file not found at {file_path}")
    return pd.read_csv(file_path)


def load_transposed_dataset(file_path):
    """
    Loads a pre-processed (transposed) dataset.
    Sets the first column (Sample Names) as the index.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: Dataset not found at {file_path}")

    return pd.read_csv(file_path, index_col=0)


def reshape_dataset(df_raw):
    """
    Transposes the raw dataset from (Features x Samples) to (Samples x Features) format.
    Extracts class labels and ensures numeric data integrity.

    Args:
        df_raw (pd.DataFrame): Raw dataframe where rows are metabolites and columns are samples.

    Returns:
        pd.DataFrame: Reshaped dataframe with 'Class' column and numeric features.
    """
    # Extract metadata components
    samples = df_raw.columns[1:]
    classes = df_raw.iloc[0, 1:].values
    metabolites = df_raw.iloc[1:, 0].values

    # Transpose the data matrix
    data_matrix = df_raw.iloc[1:, 1:].values.T

    # Reconstruct DataFrame
    df_reshaped = pd.DataFrame(data_matrix, index=samples, columns=metabolites)
    df_reshaped = df_reshaped.apply(pd.to_numeric, errors='coerce')

    # Insert Class column at the beginning
    df_reshaped.insert(0, 'Class', classes)

    return df_reshaped


def remove_qc_and_technical_replicates(df):
    """
    Filters out Quality Control (QC) samples and technical replicates from the dataset.

    Criteria:
    1. Class is 'QC'.
    2. Sample name contains the suffix '_01' (indicating a technical duplicate).

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    # 1. Remove QC samples
    df_clean = df[df['Class'] != 'QC'].copy()

    # 2. Remove technical replicates
    # regex=False ensures literal string matching for '_01'
    df_clean = df_clean[~df_clean.index.str.contains('_01', regex=False)]

    return df_clean