import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def run_svm_model(X_train, y_train, X_test, kernel='linear', C=1.0, gamma='scale', class_weight=None):
    """
    Trains a Support Vector Machine (SVM) classifier and predicts on the test set.

    Theoretical Background:
    - SVM belongs to "Kernel Based Methods".
    - Objective: Find a hyperplane that maximizes the margin (Maximal Margin Classifier)
      between classes. The samples on the margin are 'Support Vectors'.
    - Soft Margin: Controlled by parameter 'C'. Allows some misclassification to handle
      noisy data or non-separable classes (Bias-Variance trade-off).
    - Kernel Trick: Maps data to higher-dimensional space to handle non-linear boundaries.
      Common kernels: 'linear' (good for omics p >> n), 'rbf', 'poly'.

    Args:
        X_train (pd.DataFrame or np.array): Training features.
        y_train (pd.Series or np.array): Training labels.
        X_test (pd.DataFrame or np.array): Test features.
        kernel (str): 'linear', 'rbf', 'poly', etc. Default is 'linear' as it is often
                      robust for high-dimensional omics data.
        C (float): Regularization parameter.
                   - Small C: Wider margin, more violations allowed (High Bias, Low Variance).
                   - Large C: Narrower margin, strict classification (Low Bias, High Variance).
        gamma (str or float): Kernel coefficient for 'rbf', 'poly'.
                              'scale' uses 1 / (n_features * X.var()).
        class_weight (dict or 'balanced'): Set to 'balanced' to handle class imbalance
                                           (common in medical datasets) by adjusting weights
                                           inversely proportional to class frequencies.

    Returns:
        tuple: (y_pred, y_prob)
            - y_pred: Predicted class labels for X_test.
            - y_prob: Probability estimates for the positive class (useful for AUC).
    """

    # Initialize the Support Vector Classifier
    # We set probability=True to enable probability estimation (needed for AUC/ROC).
    # Note: SVMs do not natively output probabilities; sklearn uses Platt scaling (expensive).
    model = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        class_weight=class_weight,
        probability=True,
        random_state=42  # Ensure reproducibility
    )

    # 1. Training Phase
    # The algorithm solves the quadratic optimization problem to find the optimal hyperplane.
    model.fit(X_train, y_train)

    # 2. Prediction Phase
    # Predict class labels based on the sign of the discriminant function D(x).
    y_pred = model.predict(X_test)

    # Predict class probabilities (specifically for the positive class, index 1)
    # Essential for calculating ROC curves and AUC later.
    y_prob = model.predict_proba(X_test)[:, 1]

    return y_pred, y_prob


def run_logistic_regression(X_train, y_train, X_test, penalty='l2', C=1.0, solver='liblinear', class_weight=None, l1_ratio=None):
    """
    Trains a Logistic Regression model for classification and optional feature selection.

    Theoretical Background:
    - Probabilistic Nature: Uses the Sigmoid function to map output to [0, 1].
      Formula: f(X) = e^(beta*X) / (1 + e^(beta*X)).
    - Decision Boundary: Linear in terms of Log-Odds (Logit).
    - Optimization: Uses Maximum Likelihood Estimation (MLE) unlike OLS in linear regression.
    - High-Dimensionality (Omics context):
      Standard LR fails when p >> n. Regularization is essential.
      - L2 (Ridge): Shrinks coefficients (High Bias/Low Variance), good for correlation handling.
      - L1 (Lasso): Shrinks coefficients to ZERO. Acts as Embedded Feature Selection (Sparse model).
      - Elastic Net: Combination of L1 and L2.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_test (pd.DataFrame): Test features.
        penalty (str): 'l1', 'l2', 'elasticnet', or 'none'.
                       Use 'l1' for Feature Selection (Lasso).
        C (float): Inverse of regularization strength.
                   Smaller values specify stronger regularization (more sparsity in L1).
        solver (str): Algorithm to use in the optimization problem.
                      'liblinear' is good for small datasets and supports 'l1' and 'l2'.
                      'saga' is required for 'elasticnet'.
        class_weight (dict or 'balanced'): Handling class imbalance.
        l1_ratio (float): The Elastic Net mixing parameter (0 <= l1_ratio <= 1).
                          Only used if penalty='elasticnet'.

    Returns:
        tuple: (y_pred, y_prob, feature_info)
            - y_pred: Predicted class labels.
            - y_prob: Probability estimates for the positive class.
            - feature_info: A dictionary containing:
                - 'coefficients': Raw beta values.
                - 'selected_features': List of features with non-zero coefficients.
                - 'excluded_features': List of features with zero coefficients (removed by L1).
    """

    # Initialize Logistic Regression with provided hyperparameters
    model = LogisticRegression(
        penalty=penalty,
        C=C,
        solver=solver,
        class_weight=class_weight,
        l1_ratio=l1_ratio,  # Used only if penalty='elasticnet'
        max_iter=1000,  # Increased iterations to ensure convergence for high-dim data
        random_state=42
    )

    # 1. Training Phase (MLE)
    model.fit(X_train, y_train)

    # 2. Prediction Phase
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability of positive class

    # 3. Feature Selection Logic (Extraction of beta coefficients)
    # model.coef_ is an array of shape (1, n_features) for binary classification
    coefs = model.coef_[0]

    feature_info = {
        'coefficients': coefs,
        'selected_features': [],
        'excluded_features': [],
        'feature_importance_ranking': {}
    }

    # Identify which features were selected (non-zero coefficient) vs excluded (zero coefficient)
    # This is particularly relevant when penalty='l1' (Lasso) is used.
    if isinstance(X_train, pd.DataFrame):
        feature_names = X_train.columns

        # 1. Selection/Exclusion Lists
        mask = coefs != 0
        feature_info['selected_features'] = feature_names[mask].tolist()
        feature_info['excluded_features'] = feature_names[~mask].tolist()

        # 2. Importance Ranking (based on absolute value of coefficients)
        # We rank by magnitude (strength of influence), but keep the sign in the value
        # to understand direction (positive or negative correlation).
        coef_dict = dict(zip(feature_names, coefs))
        sorted_coefs = dict(sorted(coef_dict.items(), key=lambda item: abs(item[1]), reverse=True))
        feature_info['feature_importance_ranking'] = sorted_coefs

    else:
        # Fallback if input is numpy array (return indices instead of names)
        feature_info['selected_features'] = np.where(coefs != 0)[0].tolist()
        feature_info['excluded_features'] = np.where(coefs == 0)[0].tolist()
        # Fallback ranking by index
        coef_dict = dict(enumerate(coefs))
        sorted_coefs = dict(sorted(coef_dict.items(), key=lambda item: abs(item[1]), reverse=True))
        feature_info['feature_importance_ranking'] = sorted_coefs

    return y_pred, y_prob, feature_info



def run_random_forest(X_train, y_train, X_test, n_estimators=100, max_depth=None, class_weight=None):
    """
    Trains a Random Forest Classifier and extracts feature importance.

    Theoretical Background:
    - Ensemble Method: Uses Bagging (Bootstrap Aggregating) to reduce variance and overfitting
      compared to single Decision Trees.
    - Randomness:
        1. Data Sampling: Each tree is trained on a bootstrap sample (with replacement).
        2. Feature Sampling: At each split, only a random subset of features (usually sqrt(p))
           is considered. This de-correlates the trees.
    - Robustness: Highly suitable for omics data (High Dimensionality, p >> n).
    - Interpretability: While the ensemble is a "black box" compared to a single tree,
      it provides 'Feature Importance' based on the average reduction in impurity (e.g., Gini).

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_test (pd.DataFrame): Test features.
        n_estimators (int): Number of trees in the forest.
                            More trees = more robust (less variance), but higher computational cost.
        max_depth (int): Maximum depth of the tree. None means nodes are expanded until all leaves
                         are pure or contain less than min_samples_split samples.
        class_weight (dict or 'balanced'): Weights associated with classes to handle imbalance.
                                           'balanced' uses the values of y to automatically adjust
                                           weights inversely proportional to class frequencies.

    Returns:
        tuple: (y_pred, y_prob, feature_info)
            - y_pred: Predicted class labels.
            - y_prob: Probability estimates for the positive class.
            - feature_info: A dictionary containing:
                - 'importances': Raw importance scores (sum to 1).
                - 'feature_importance_ranking': Dictionary mapping feature names to their scores,
                                                sorted descending.
    """

    # Initialize Random Forest Classifier
    # n_jobs=-1 allows using all processors to speed up training
    # max_features='sqrt' is the standard for classification as per theory
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features='sqrt',
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )

    # 1. Training Phase (Bagging)
    model.fit(X_train, y_train)

    # 2. Prediction Phase (Majority Voting)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability for positive class

    # 3. Feature Importance Extraction (Gini Impurity Reduction)
    importances = model.feature_importances_

    feature_info = {
        'importances': importances,
        'feature_importance_ranking': {}
    }

    # Map importances to feature names if available and sort them
    if isinstance(X_train, pd.DataFrame):
        feature_names = X_train.columns
        # Create a dictionary of feature names and their importance
        importance_dict = dict(zip(feature_names, importances))
        # Sort by importance (descending)
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))
        feature_info['feature_importance_ranking'] = sorted_importance
    else:
        # Fallback for numpy arrays
        feature_info['feature_importance_ranking'] = dict(enumerate(importances))

    return y_pred, y_prob, feature_info