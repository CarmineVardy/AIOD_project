import numpy as np
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, balanced_accuracy_score,
                             roc_auc_score, matthews_corrcoef)


def compute_classification_metrics(y_true, y_pred, y_prob=None):
    """
    Computes a comprehensive set of evaluation metrics for binary classification tasks,
    specifically tailored for omics data analysis where class imbalance is common.

    Based on the theoretical framework:
    - Confusion Matrix components (TP, TN, FP, FN)
    - Threshold-based metrics (Accuracy, Sensitivity, Specificity, etc.)
    - Probabilistic metrics (AUC) if probabilities are provided.

    Args:
        y_true (array-like): Ground truth (correct) target values (e.g., [0, 1, 0...]).
        y_pred (array-like): Estimated targets as returned by a classifier (binary labels).
        y_prob (array-like, optional): Probability estimates of the positive class.
                                       Required for AUC calculation. Defaults to None.

    Returns:
        dict: A dictionary containing the calculated metrics.
    """

    # 1. Confusion Matrix Calculation
    # The confusion matrix C is such that C_{i, j} is equal to the number of observations
    # known to be in group i and predicted to be in group j.
    # In binary classification (0=Negative/Healthy, 1=Positive/CHD):
    # tn = True Negatives, fp = False Positives
    # fn = False Negatives, tp = True Positives
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # 2. Metric Calculations based on the provided formulas

    # Sensitivity (Recall / True Positive Rate)
    # Formula: TP / (TP + FN)
    # Goal: Measure ability to find all positive samples (Minimize False Negatives).
    sensitivity = recall_score(y_true, y_pred)

    # Specificity (True Negative Rate)
    # Formula: TN / (FP + TN)
    # Goal: Measure ability to identify negatives correctly (Minimize False Alarms).
    # Note: sklearn does not have a direct 'specificity' function, so we calculate it manually.
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Precision (Positive Predictive Value - PPV)
    # Formula: TP / (TP + FP)
    # Goal: Reliability of positive predictions.
    precision = precision_score(y_true, y_pred, zero_division=0)

    # Accuracy
    # Formula: (TP + TN) / Total
    # Note: Can be misleading in imbalanced datasets (e.g. 990 negatives, 10 positives).
    accuracy = accuracy_score(y_true, y_pred)

    # F1 Score
    # Formula: 2 * (Precision * Recall) / (Precision + Recall)
    # Goal: Harmonic mean, good for imbalanced data when positive class is the focus.
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Balanced Accuracy (BA)
    # Formula: (Sensitivity + Specificity) / 2
    # Goal: Arithmetic mean of TPR and TNR. Penalizes models that ignore the minority class.
    # Preferable to Accuracy in our context (CHD vs Healthy).
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    # Matthews Correlation Coefficient (MCC)
    # A holistic metric that takes into account true and false positives and negatives.
    # Generally regarded as a balanced measure which can be used even if the classes are of very different sizes.
    mcc = matthews_corrcoef(y_true, y_pred)

    # 3. Constructing the Output Dictionary
    metrics = {
        "Confusion_Matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "F1_Score": f1,
        "Balanced_Accuracy": balanced_acc,
        "MCC": mcc,
        "AUC": None  # Default to None if probabilities are not provided
    }

    # 4. ROC - Area Under Curve (AUC)
    # Requires probability estimates (y_prob), not just binary labels.
    # An AUC of 0.5 implies no discrimination (random), 1.0 implies perfect discrimination.
    if y_prob is not None:
        try:
            # We assume y_prob are probabilities of the positive class
            roc_auc = roc_auc_score(y_true, y_prob)
            metrics["AUC"] = roc_auc
        except ValueError:
            # Handle cases where AUC cannot be computed (e.g., only one class present in y_true)
            print("Warning: AUC could not be computed (possibly only one class in y_true).")
            metrics["AUC"] = np.nan

    return metrics