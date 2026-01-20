"""
Data Fusion Module.

This module implements Low-Level Data Fusion strategies for multi-block metabolomics data.
It includes functions for:
1. Block Scaling (Frobenius Norm) to ensure fair contribution from each analytical platform.
2. Concatenation of multiple datasets into a single super-matrix.
3. Rigorous split-aware scaling to prevent data leakage during model validation.
"""

import pandas as pd
import numpy as np


def _process_block(df, index):
    """
    Internal helper to apply Block Scaling and rename columns for a single dataframe.

    Methodology:
    Divides the block by its Frobenius Norm (square root of the sum of squared elements).
    This equalizes the variance (energy) across blocks, preventing one dataset
    from dominating the fusion due to scale or size.

    Args:
        df (pd.DataFrame): Input dataframe (numeric features only).
        index (int): Block index (used for column suffixing).

    Returns:
        pd.DataFrame: Scaled and renamed dataframe.
    """
    features = df.copy()

    # Calculate Frobenius Norm
    norm = np.linalg.norm(features.values, 'fro')

    if norm == 0:
        scale_factor = 1.0
    else:
        scale_factor = 1.0 / norm

    features_scaled = features * scale_factor

    # Rename columns to ensure uniqueness (e.g., "Mz_100" -> "Mz_100_Block1")
    features_scaled.columns = [f"{col}_Block{index + 1}" for col in features_scaled.columns]

    return features_scaled


def apply_block_weighting_split(X_train, X_test, block_index):
    """
    Applies Block Scaling (Frobenius Norm) preventing Data Leakage.

    Crucial Step:
    The scaling factor is calculated strictly on the TRAINING set and then applied
    to both Training and Test sets. This ensures the Test set remains unseen.

    Args:
        X_train (pd.DataFrame): Training features for the specific block.
        X_test (pd.DataFrame): Test features for the specific block.
        block_index (int): Index of the block (0 for ESI-, 1 for ESI+).

    Returns:
        tuple: (X_train_weighted, X_test_weighted) with renamed columns.
    """
    # 1. Compute Norm on TRAIN only
    norm = np.linalg.norm(X_train.values, 'fro')

    if norm == 0:
        scale_factor = 1.0
    else:
        scale_factor = 1.0 / norm

    # 2. Apply factor to both sets
    X_train_weighted = X_train * scale_factor
    X_test_weighted = X_test * scale_factor

    # 3. Rename columns to avoid collisions during concatenation
    suffix = f"_Block{block_index + 1}"
    X_train_weighted.columns = [f"{col}{suffix}" for col in X_train_weighted.columns]
    X_test_weighted.columns = [f"{col}{suffix}" for col in X_test_weighted.columns]

    return X_train_weighted, X_test_weighted


def low_level_fusion(dataframe_list):
    """
    Orchestrates standard Low-Level Data Fusion on a list of datasets.

    Workflow:
    1. Extracts the 'Class' column from the first block.
    2. Scales each block individually using Frobenius Norm.
    3. Concatenates all blocks horizontally into a single super-matrix.

    Args:
        dataframe_list (list): List of pandas DataFrames (one per block).

    Returns:
        pd.DataFrame: Fused dataset containing all features and the Class column.
    """
    if not dataframe_list or len(dataframe_list) < 2:
        return None

    scaled_blocks = []
    final_class_col = None

    for i, df in enumerate(dataframe_list):
        # Extract class only once
        if i == 0:
            final_class_col = df['Class']

        numeric_df = df.drop(columns=['Class'])

        # Apply scaling and renaming
        scaled_feat = _process_block(numeric_df, i)
        scaled_blocks.append(scaled_feat)

    # Concatenate Class + Scaled Blocks
    all_data = [final_class_col] + scaled_blocks
    pd_merged = pd.concat(all_data, axis=1)

    return pd_merged