"""
Dataset Splitting Module.

This module provides various strategies for splitting datasets into Training and Test sets,
specifically designed for Omics data analysis where Data Leakage prevention is paramount.

Strategies implemented:
1. Random K-Fold CV (Baseline)
2. Stratified K-Fold CV (Maintains class balance)
3. Duplex Algorithm (Deterministic, maximizes spatial coverage)

It also includes a safe scaling utility to fit parameters on Training data only.
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from src.pca import perform_pca


def apply_scaling(X_train, X_test):
    """
    Applies StandardScaler (Autoscaling) preventing Data Leakage.

    Methodology:
    Fits the scaler on X_train ONLY, then transforms both X_train and X_test using
    the parameters (mean, std) learned from the training set.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.

    Returns:
        tuple: (X_train_scaled, X_test_scaled) preserving DataFrame structure.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train), index=X_train.index, columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), index=X_test.index, columns=X_test.columns
    )

    return X_train_scaled, X_test_scaled


def get_random_kfold_cv(df, target_col='Class', n_splits=5, random_state=42, perform_scaling=True):
    """
    Implements Random K-Fold Cross-Validation.

    Used as a baseline to demonstrate the potential instability of random splits compared
    to stratified approaches in imbalanced clinical datasets.

    Args:
        df (pd.DataFrame): Input dataset.
        target_col (str): Target class column.
        n_splits (int): Number of folds.
        perform_scaling (bool): If True, applies autoscaling within each fold.

    Returns:
        list: A list of tuples [(X_train, X_test, y_train, y_test), ...].
    """
    print(f"--- Applying Random (Unstratified) {n_splits}-Fold CV ---")

    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col].values

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds_data = []

    for train_idx, test_idx in kf.split(X):
        X_train_fold = X.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_train_fold = y[train_idx]
        y_test_fold = y[test_idx]

        if perform_scaling:
            X_train_fold, X_test_fold = apply_scaling(X_train_fold, X_test_fold)

        folds_data.append((X_train_fold, X_test_fold, y_train_fold, y_test_fold))

    return folds_data


def get_stratified_kfold_cv(df, target_col='Class', n_splits=5, random_state=42, perform_scaling=True):
    """
    Implements Stratified K-Fold Cross-Validation.

    Best Practice for Omics: Ensures that the proportion of biological classes (e.g., CHD vs CTRL)
    is preserved in every fold, preventing bias during model training.

    Args:
        df (pd.DataFrame): Input dataset.
        n_splits (int): Number of folds.

    Returns:
        list: List of fold tuples.
    """
    print(f"--- Applying Stratified {n_splits}-Fold CV ---")

    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds_data = []

    for train_idx, test_idx in skf.split(X, y):
        X_train_fold = X.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_train_fold = y[train_idx]
        y_test_fold = y[test_idx]

        if perform_scaling:
            X_train_fold, X_test_fold = apply_scaling(X_train_fold, X_test_fold)

        folds_data.append((X_train_fold, X_test_fold, y_train_fold, y_test_fold))

    return folds_data


def get_duplex_split(df, target_col='Class', split_ratio=0.75, use_pca=True, n_pc=None, perform_scaling=True):
    """
    Implements the Duplex algorithm for deterministic dataset splitting.

    Methodology:
    Iteratively selects the two most distant samples in the dataset and assigns them
    alternatingly to the Training and Test sets. This ensures the Test set covers
    the full experimental space (interpolation + extrapolation) and is not just a subset
    of the mean distribution.

    Args:
        df (pd.DataFrame): Input dataset.
        split_ratio (float): Fraction of samples for Training.
        use_pca (bool): If True, calculates Euclidean distance on PCA Scores (recommended for high-dim data).
        n_pc (int/float): Number of PCs or variance ratio for PCA.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print(f"--- Applying Duplex Split (Deterministic & Balanced Coverage) ---")

    X_raw = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col].values
    n_samples = X_raw.shape[0]

    n_train_target = int(n_samples * split_ratio)
    n_test_target = n_samples - n_train_target

    # 1. Define Geometry Space
    if use_pca:
        # Calculate distances in latent space to avoid curse of dimensionality
        pca_results = perform_pca(df, n_components=n_pc, scaling=None)
        X_geometry = pca_results['scores'].values
    else:
        # Calculate distances in raw scaled space
        X_geometry = StandardScaler().fit_transform(X_raw)

    # 2. Compute Distance Matrix
    dist_matrix = cdist(X_geometry, X_geometry, metric='euclidean')

    remaining_indices = list(range(n_samples))
    train_indices = []
    test_indices = []

    # 3. Iterative Selection Loop
    while len(remaining_indices) > 0:

        # --- Phase A: Select for Training Set ---
        if len(train_indices) < n_train_target:
            if len(train_indices) == 0:
                # Initialization: Find the 2 most distant points globally
                sub_dist = dist_matrix[np.ix_(remaining_indices, remaining_indices)]
                i, j = np.unravel_index(np.argmax(sub_dist), sub_dist.shape)
                global_i, global_j = remaining_indices[i], remaining_indices[j]

                train_indices.extend([global_i, global_j])
                if global_i in remaining_indices: remaining_indices.remove(global_i)
                if global_j in remaining_indices: remaining_indices.remove(global_j)
            else:
                # Find point furthest from existing Train set (Maximin)
                dist_to_train = dist_matrix[np.ix_(remaining_indices, train_indices)]
                min_dists = np.min(dist_to_train, axis=1)
                best_local_idx = np.argmax(min_dists)

                best_global_idx = remaining_indices[best_local_idx]
                train_indices.append(best_global_idx)
                remaining_indices.pop(best_local_idx)

        # --- Phase B: Select for Test Set ---
        if len(remaining_indices) > 0 and len(test_indices) < n_test_target:
            if len(test_indices) == 0:
                # Initialization for Test set from remaining points
                sub_dist = dist_matrix[np.ix_(remaining_indices, remaining_indices)]
                i, j = np.unravel_index(np.argmax(sub_dist), sub_dist.shape)
                global_i, global_j = remaining_indices[i], remaining_indices[j]

                test_indices.extend([global_i, global_j])
                if global_i in remaining_indices: remaining_indices.remove(global_i)
                if global_j in remaining_indices: remaining_indices.remove(global_j)
            else:
                # Find point furthest from existing Test set
                dist_to_test = dist_matrix[np.ix_(remaining_indices, test_indices)]
                min_dists = np.min(dist_to_test, axis=1)
                best_local_idx = np.argmax(min_dists)

                best_global_idx = remaining_indices[best_local_idx]
                test_indices.append(best_global_idx)
                remaining_indices.pop(best_local_idx)

    # 4. Final Construction
    X_train = X_raw.iloc[train_indices]
    X_test = X_raw.iloc[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    if perform_scaling:
        X_train, X_test = apply_scaling(X_train, X_test)

    return X_train, X_test, y_train, y_test