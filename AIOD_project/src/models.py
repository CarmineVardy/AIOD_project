"""
Predictive Modeling and Metrics Module.

This module provides wrappers for training standard Machine Learning classifiers
(Logistic Regression, SVM, Random Forest) in the context of Omics data.
It includes specific logic for:
1. Handling class imbalance (class_weight).
2. Extracting feature importance/coefficients.
3. Calculating comprehensive classification metrics (Sensitivity, Specificity, AUC).
4. Evaluating the quality of data splits (Spatial coverage, Class balance).
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, recall_score, precision_score, accuracy_score,
    balanced_accuracy_score, roc_auc_score
)
from sklearn.svm import SVC


def run_logistic_regression(X_train, y_train, X_test, penalty='l2', C=1.0, solver='liblinear', class_weight=None,
                            l1_ratio=None):
    """
    Trains a Logistic Regression model.
    Supports L1 (Lasso) regularization for embedded feature selection.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_test (pd.DataFrame): Test features.
        penalty (str): 'l1', 'l2', 'elasticnet', or 'none'.
        C (float): Inverse of regularization strength.
        solver (str): Algorithm for optimization ('liblinear', 'saga').
        class_weight (dict or 'balanced'): Handling class imbalance.

    Returns:
        tuple: (y_pred, y_prob, feature_info)
    """
    model = LogisticRegression(
        penalty=penalty,
        C=C,
        solver=solver,
        class_weight=class_weight,
        l1_ratio=l1_ratio,
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    pos_class_idx = np.where(model.classes_ == 'CHD')[0][0]
    y_prob = model.predict_proba(X_test)[:, pos_class_idx]

    # Feature Importance Extraction
    coefs = model.coef_[0]
    feature_names = X_train.columns
    mask = coefs != 0

    feature_info = {
        'coefficients': coefs,
        'selected_features': feature_names[mask].tolist(),
        'excluded_features': feature_names[~mask].tolist(),
        'feature_importance_ranking': dict(
            sorted(zip(feature_names, coefs), key=lambda item: abs(item[1]), reverse=True)
        )
    }

    return y_pred, y_prob, feature_info


def run_svm_model(X_train, y_train, X_test, kernel='linear', C=1.0, gamma='scale', class_weight=None):
    """
    Trains a Support Vector Machine (SVM) classifier.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_test (pd.DataFrame): Test features.
        kernel (str): 'linear', 'rbf', 'poly'.
        C (float): Regularization parameter.
        gamma (str/float): Kernel coefficient.

    Returns:
        tuple: (y_pred, y_prob, feature_info)
    """
    model = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        probability=True,
        class_weight=class_weight,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    pos_class_idx = np.where(model.classes_ == 'CHD')[0][0]
    y_prob = model.predict_proba(X_test)[:, pos_class_idx]

    feature_info = {'coefficients': None, 'feature_importance_ranking': {}}

    if kernel == 'linear':
        coefs = model.coef_[0]
        feature_info['coefficients'] = coefs
        feature_info['feature_importance_ranking'] = dict(
            sorted(zip(X_train.columns, coefs), key=lambda item: abs(item[1]), reverse=True)
        )

    return y_pred, y_prob, feature_info


def run_random_forest(X_train, y_train, X_test, n_estimators=100, max_depth=None, class_weight=None):
    """
    Trains a Random Forest Classifier using Bagging and Feature Randomness.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_test (pd.DataFrame): Test features.
        n_estimators (int): Number of trees.
        max_depth (int): Maximum depth of the tree.

    Returns:
        tuple: (y_pred, y_prob, feature_info)
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features='sqrt',
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    pos_class_idx = np.where(model.classes_ == 'CHD')[0][0]
    y_prob = model.predict_proba(X_test)[:, pos_class_idx]

    importances = model.feature_importances_
    feature_info = {
        'importances': importances,
        'feature_importance_ranking': dict(
            sorted(zip(X_train.columns, importances), key=lambda item: item[1], reverse=True)
        )
    }

    return y_pred, y_prob, feature_info


def compute_classification_metrics(y_true, y_pred, y_prob, pos_label='CHD', neg_label='CTRL',
                                   model_name="Generic Model"):
    """
    Computes classification metrics including Accuracy, Sensitivity, Specificity, and AUC.
    Enforces correct label mapping (Positive=CHD, Negative=CTRL).

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        y_prob (array-like): Probability of positive class.

    Returns:
        dict: Calculated metrics.
    """
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[neg_label, pos_label]).ravel()
    except ValueError:
        tn, fp, fn, tp = 0, 0, 0, 0

    sensitivity = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    y_true_binary = (np.array(y_true) == pos_label).astype(int)
    auc_value = roc_auc_score(y_true_binary, y_prob)

    return {
        "Model_Name": model_name,
        "Accuracy": accuracy,
        "Balanced_Accuracy": balanced_acc,
        "AUC": auc_value,
        "Classification_Error": 1.0 - accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)
    }


def evaluate_split_quality(X_train, X_test, y_train, y_test, split_name="Generic Split"):
    """
    Evaluates statistical and spatial characteristics of a train/test split.
    Useful to detect class imbalance or distribution shifts.

    Args:
        X_train, X_test (pd.DataFrame): Feature matrices.
        y_train, y_test (np.array): Target labels.

    Returns:
        dict: Metrics including Class Balance Discrepancy and Spatial Distance Ratio.
    """
    metrics = {"Split_Strategy": split_name}

    # 1. Class Balance
    train_counts = pd.Series(y_train).value_counts(normalize=True) * 100
    test_counts = pd.Series(y_test).value_counts(normalize=True) * 100

    balance_diff = (train_counts - test_counts).abs().sum()
    metrics['Class_Balance_Discrepancy'] = balance_diff

    all_classes = set(train_counts.index).union(set(test_counts.index))
    for cls in all_classes:
        metrics[f"Train_Prop_{cls}"] = train_counts.get(cls, 0.0)
        metrics[f"Test_Prop_{cls}"] = test_counts.get(cls, 0.0)
        metrics[f"Train_Count_{cls}"] = int(pd.Series(y_train).value_counts().get(cls, 0))
        metrics[f"Test_Count_{cls}"] = int(pd.Series(y_test).value_counts().get(cls, 0))

    # 2. Global Stats
    metrics['Train_Global_Mean'] = X_train.values.mean()
    metrics['Train_Global_Std'] = X_train.values.std()
    metrics['Test_Global_Mean'] = X_test.values.mean()
    metrics['Test_Global_Std'] = X_test.values.std()

    # 3. Spatial Coverage (Centroid Distance)
    X_all = pd.concat([X_train, X_test])
    centroid = X_all.mean().values.reshape(1, -1)

    dist_train = cdist(X_train, centroid, metric='euclidean').flatten()
    dist_test = cdist(X_test, centroid, metric='euclidean').flatten()

    avg_dist_train = np.mean(dist_train)
    avg_dist_test = np.mean(dist_test)

    metrics['Train_Avg_Spatial_Dist'] = avg_dist_train
    metrics['Test_Avg_Spatial_Dist'] = avg_dist_test
    metrics['Spatial_Dist_Ratio'] = avg_dist_test / avg_dist_train if avg_dist_train > 0 else 0

    return metrics