import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.model_selection import (
    KFold,
    StratifiedKFold
)

from src.pca import *


def apply_scaling(X_train, X_test):
    """
    Applies StandardScaler (Autoscaling) to the datasets.

    Data Leakage Prevention:
    - Fits the scaler ONLY on X_train.
    - Transforms both X_train and X_test using the parameters learned from X_train.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.

    Returns:
        tuple: (X_train_scaled, X_test_scaled) as pandas DataFrames.
    """
    scaler = StandardScaler()

    # Fit on Training Data ONLY
    scaler.fit(X_train)

    # Transform both sets and preserve DataFrame structure
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train),
        index=X_train.index,
        columns=X_train.columns
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        index=X_test.index,
        columns=X_test.columns
    )

    return X_train_scaled, X_test_scaled


def get_random_kfold_cv(df, target_col='Class', n_splits=5, random_state=42, perform_scaling=True):
    """
    Implements Random K-Fold Cross-Validation (No Stratification).

    Theoretical Context:
    - Pure random assignment to folds.
    - Does NOT check class balance (KFold instead of StratifiedKFold).
    - Useful to demonstrate instability/bias compared to Stratified CV.

    Args:
        df (pd.DataFrame): Input dataset.
        target_col (str): Target column.
        n_splits (int): Number of folds (k).
        random_state (int): Seed for reproducibility.
        perform_scaling (bool): Apply Autoscaling inside each fold.

    Returns:
        list: A list of tuples [(X_train, X_test, y_train, y_test), ...].
    """
    print(f"--- Applying Random (Unstratified) {n_splits}-Fold CV ---")

    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col].values

    # Standard KFold with shuffle=True is strictly random
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds_data = []

    # Note: kf.split(X) does not use 'y' to split, respecting the "No Class Constraint" theory
    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
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

    Theoretical Context:
    - Best practice for Omics/Imbalanced datasets.
    - Splits data into 'k' folds ensuring class proportions are preserved in each fold.

    Args:
        df (pd.DataFrame): Input dataset.
        target_col (str): Target column.
        n_splits (int): Number of folds (k).
        random_state (int): Seed for reproducibility.
        perform_scaling (bool): Apply Autoscaling inside each fold.

    Returns:
        list: A list of tuples [(X_train, X_test, y_train, y_test), ...].
    """
    print(f"--- Applying Stratified {n_splits}-Fold CV ---")

    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col].values

    # StratifiedKFold enforces class balance
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds_data = []

    for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
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

    Theoretical Context:
    - An evolution of Kennard-Stone (KS).
    - Goal: Construct Training and Test sets with statistically equivalent coverage
      of the experimental space.
    - Mechanism: Alternating Maximin selection. Unlike KS (which fills Train first),
      Duplex assigns distant samples to BOTH sets iteratively.
    - Advantage: Provides more realistic performance estimates than KS because the Test set
      is not confined inside the Training set's convex hull.

    PCA Integration:
    - Strongly recommended for high-dimensional Omics data to avoid the "Curse of Dimensionality".
    - Distances are calculated on PCA Scores.

    Args:
        df (pd.DataFrame): Complete input dataset.
        target_col (str): Name of the target column.
        split_ratio (float): Ratio of samples for Training (default 0.75).
        use_pca (bool): If True, computes distances on PCA Scores.
        n_pc (int): Number of PCs to use. If None, automatic selection.
        perform_scaling (bool): Whether to apply Autoscaling on the FINAL split output.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print(f"--- Applying Duplex Split (Deterministic & Balanced Coverage) ---")

    # 1. Preparation of X and y (Raw Data)
    X_raw = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col].values
    n_samples = X_raw.shape[0]

    # Calculate target sizes
    n_train_target = int(n_samples * split_ratio)
    n_test_target = n_samples - n_train_target

    print(f"Total Samples: {n_samples} | Target Train: {n_train_target} | Target Test: {n_test_target}")

    # 2. Geometry Definition (PCA vs Raw)
    if use_pca:
        print("Duplex Geometry: Using PCA Scores (High-Dimensionality handling)...")
        # Reuse existing PCA logic
        pca_results = perform_pca(df, n_components=n_pc, scaling='autoscaling')
        X_geometry = pca_results['scores'].values
        print(f"PCA Space Dimensions: {X_geometry.shape[1]} PCs")
    else:
        print("Duplex Geometry: Using Raw Variables (Scaled)...")
        from sklearn.preprocessing import StandardScaler
        X_geometry = StandardScaler().fit_transform(X_raw)

    # 3. Duplex Algorithm Execution
    # Full distance matrix
    dist_matrix = cdist(X_geometry, X_geometry, metric='euclidean')

    remaining_indices = list(range(n_samples))
    train_indices = []
    test_indices = []

    # Iteration loop
    # We continue until we run out of samples or fill both sets
    while len(remaining_indices) > 0:

        # --- PHASE A: SELECT FOR TRAINING SET ---
        if len(train_indices) < n_train_target:
            if len(train_indices) == 0:
                # Initialization: Find the 2 most distant points in the whole dataset
                # Argmax over the sub-matrix of remaining points
                sub_dist = dist_matrix[np.ix_(remaining_indices, remaining_indices)]
                i, j = np.unravel_index(np.argmax(sub_dist), sub_dist.shape)

                # Convert local sub-matrix indices to global indices
                global_i, global_j = remaining_indices[i], remaining_indices[j]

                train_indices.extend([global_i, global_j])
                # Remove carefully
                if global_i in remaining_indices: remaining_indices.remove(global_i)
                if global_j in remaining_indices: remaining_indices.remove(global_j)
            else:
                # Iterative: Find point furthest from EXISTING Train set
                # Distances between Remaining candidates (rows) and Current Train (cols)
                dist_to_train = dist_matrix[np.ix_(remaining_indices, train_indices)]

                # Min distance to any point in Train
                min_dists = np.min(dist_to_train, axis=1)
                # Max of Min (Maximin)
                best_local_idx = np.argmax(min_dists)

                best_global_idx = remaining_indices[best_local_idx]
                train_indices.append(best_global_idx)
                remaining_indices.pop(best_local_idx)

        # --- PHASE B: SELECT FOR TEST SET ---
        if len(remaining_indices) > 0 and len(test_indices) < n_test_target:
            if len(test_indices) == 0:
                # Initialization: Find 2 most distant points among REMAINING
                sub_dist = dist_matrix[np.ix_(remaining_indices, remaining_indices)]
                i, j = np.unravel_index(np.argmax(sub_dist), sub_dist.shape)

                global_i, global_j = remaining_indices[i], remaining_indices[j]

                test_indices.extend([global_i, global_j])
                if global_i in remaining_indices: remaining_indices.remove(global_i)
                if global_j in remaining_indices: remaining_indices.remove(global_j)
            else:
                # Iterative: Find point furthest from EXISTING Test set
                dist_to_test = dist_matrix[np.ix_(remaining_indices, test_indices)]

                min_dists = np.min(dist_to_test, axis=1)
                best_local_idx = np.argmax(min_dists)

                best_global_idx = remaining_indices[best_local_idx]
                test_indices.append(best_global_idx)
                remaining_indices.pop(best_local_idx)

    print(f"Duplex Selection Complete. Train: {len(train_indices)}, Test: {len(test_indices)}")

    # 4. Construct Final Sets (Using Raw Features)
    X_train = X_raw.iloc[train_indices]
    X_test = X_raw.iloc[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    # 5. Final Scaling (Anti-Leakage)
    if perform_scaling:
        print("-> Applying Autoscaling on Final Sets...")
        X_train, X_test = apply_scaling(X_train, X_test)

    return X_train, X_test, y_train, y_test