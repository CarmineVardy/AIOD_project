import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import (
    confusion_matrix, recall_score, precision_score, accuracy_score,
    balanced_accuracy_score
)

def compute_classification_metrics(y_true, y_pred, pos_label='CHD', neg_label='CTRL', model_name="Generic Model"):
    """
    Computes a comprehensive set of evaluation metrics for binary classification tasks.
    It strictly enforces the Confusion Matrix structure based on provided labels.

    Arguments:
        y_true (array-like): Ground truth target values (containing pos_label and neg_label).
        y_pred (array-like): Estimated targets as returned by a classifier.
        y_prob (array-like, optional): Probability estimates of the POSITIVE class.
                                       Required for AUC calculation.
        pos_label (str): The label considered as Positive (e.g., 'CHD').
        neg_label (str): The label considered as Negative (e.g., 'CTRL').

    Returns:
        dict: A flat dictionary containing the calculated metrics.
    """
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[neg_label, pos_label]).ravel()
    except ValueError:
        tn, fp, fn, tp = 0, 0, 0, 0

    # Sensitivity (Recall / TPR) -> TP / (TP + FN)
    sensitivity = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)

    # Specificity (TNR) -> TN / (FP + TN)
    # Calculated manually as sklearn does not have a direct specificity function
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Precision (PPV) -> TP / (TP + FP)
    precision = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)

    # Accuracy -> (TP + TN) / Total
    accuracy = accuracy_score(y_true, y_pred)

    # Classification Error -> 1 - Accuracy
    class_error = 1.0 - accuracy

    # Balanced Accuracy -> Arithmetic mean of Sensitivity and Specificity
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    metrics = {
        "Model_Name": model_name,
        "Accuracy": accuracy,
        "Balanced_Accuracy": balanced_acc,
        "Classification_Error": class_error,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn)
    }

    return metrics

def evaluate_split_quality(X_train, X_test, y_train, y_test, split_name="Generic Split"):
    """
    Evaluates the quality of a Train/Test split using statistical and spatial metrics.

    Returns a flat dictionary suitable for CSV export (no nested dictionaries).

    Metrics Calculated:
    - Class Balance Discrepancy (Sum of absolute diffs in class %).
    - Global Statistics (Mean & Std for Train and Test).
    - Spatial Coverage (Avg Euclidean distance from centroid).

    Args:
        X_train, X_test (pd.DataFrame): Feature matrices (Numeric only).
        y_train, y_test (np.array): Target labels.
        split_name (str): Name of the split strategy (e.g., "Stratified_Fold1").

    Returns:
        dict: A flat dictionary containing all calculated metrics.
    """
    metrics = {
        "Split_Strategy": split_name
    }

    # --- 1. Class Balance (Proportions) ---
    # Calculate percentages
    train_counts = pd.Series(y_train).value_counts(normalize=True) * 100
    test_counts = pd.Series(y_test).value_counts(normalize=True) * 100

    train_counts_raw = pd.Series(y_train).value_counts()
    test_counts_raw = pd.Series(y_test).value_counts()

    # Calculate total discrepancy
    balance_diff = (train_counts - test_counts).abs().sum()
    metrics['Class_Balance_Discrepancy'] = balance_diff

    # Add specific class percentages to the flat dict (e.g., Train_Prop_CTRL)
    # This ensures columns in CSV are like "Train_Prop_CTRL", "Train_Prop_CHD"
    all_classes = set(train_counts.index).union(set(test_counts.index))
    for cls in all_classes:
        metrics[f"Train_Prop_{cls}"] = train_counts.get(cls, 0.0)
        metrics[f"Test_Prop_{cls}"] = test_counts.get(cls, 0.0)
        metrics[f"Train_Count_{cls}"] = int(train_counts_raw.get(cls, 0))
        metrics[f"Test_Count_{cls}"] = int(test_counts_raw.get(cls, 0))

    # --- 2. Global Statistics (Mean & Variance) ---
    # Convert to values to ensure calculation on the whole matrix
    metrics['Train_Global_Mean'] = X_train.values.mean()
    metrics['Train_Global_Std'] = X_train.values.std()
    metrics['Test_Global_Mean'] = X_test.values.mean()
    metrics['Test_Global_Std'] = X_test.values.std()

    # --- 3. Spatial Coverage (Centroid Distance) ---
    # Calculate Global Centroid
    X_all = pd.concat([X_train, X_test])
    centroid = X_all.mean().values.reshape(1, -1)

    # Calculate Euclidean distances
    dist_train = cdist(X_train, centroid, metric='euclidean').flatten()
    dist_test = cdist(X_test, centroid, metric='euclidean').flatten()

    avg_dist_train = np.mean(dist_train)
    avg_dist_test = np.mean(dist_test)

    metrics['Train_Avg_Spatial_Dist'] = avg_dist_train
    metrics['Test_Avg_Spatial_Dist'] = avg_dist_test
    metrics['Spatial_Dist_Ratio'] = avg_dist_test / avg_dist_train if avg_dist_train > 0 else 0

    return metrics