import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, 
    StratifiedKFold, 
    LeaveOneOut, 
    GroupShuffleSplit,
    StratifiedGroupKFold,
    LeaveOneGroupOut
)
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.metrics import accuracy_score

class DatasetSplitting:
    """
    A class to handle dataset splitting strategies, specifically designed to respect 
    biological replicates and sample independence (e.g., samples from the same mother 
    must stay in the same set).

    Attributes:
    -----------
    df : pd.DataFrame
        The dataset containing features and metadata.
    target_col : str
        The name of the column containing the target labels (y).
    group_col : str, optional
        The name of the column identifying biological groups (e.g., 'Mother_ID').
        If provided, all splits will respect group integrity.
    """

    def __init__(self, df, target_col, group_col=None):
        self.df = df
        self.target_col = target_col
        self.group_col = group_col
        
        # 1. Define columns to exclude from Features (Target + Group)
        cols_to_drop = [target_col]
        if group_col:
            cols_to_drop.append(group_col)
            self.groups = df[group_col].values
        else:
            self.groups = None
            
        # 2. Extract X (Features)
        # We drop the metadata columns AND strictly select only numeric columns.
        # This prevents string columns (like Sample Names) from crashing the ML model.
        self.X = df.drop(columns=cols_to_drop).select_dtypes(include=[np.number])
        
        # 3. Extract y (Labels)
        self.y = df[target_col].values
        
        # Store indices for visualization
        self.splits_history = {}

    def split_train_test(self, test_size=0.3, random_state=42):
        """
        Performs a simple Train/Test split (e.g., 70/30 or 80/20).
        Resects biological groups if 'group_col' is provided.
        """
        if self.groups is not None:
            print(f"Applying Group-wise Split (Test Size: {test_size}) respecting '{self.group_col}'...")
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(splitter.split(self.X, self.y, self.groups))
            
            X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
        else:
            print(f"Applying Standard Stratified Split (Test Size: {test_size})...")
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=test_size, stratify=self.y, random_state=random_state
            )
        
        # Save for visualization (indices only)
        train_indices = X_train.index.to_numpy()
        test_indices = X_test.index.to_numpy()
        self.splits_history['Train/Test Split'] = [(self._get_positional_indices(train_indices), 
                                                    self._get_positional_indices(test_indices))]
        
        return X_train, X_test, y_train, y_test

    def leave_one_out(self):
        """
        Performs Leave-One-Out (LOO) or Leave-One-Group-Out (LOGO).
        """
        if self.groups is not None:
            print(f"Detected biological groups. Switching LOO to Leave-One-Group-Out ({self.group_col})")
            cv = LeaveOneGroupOut()
            split_gen = cv.split(self.X, self.y, self.groups)
        else:
            print("Applying standard Leave-One-Out.")
            cv = LeaveOneOut()
            split_gen = cv.split(self.X, self.y)
            
        # Store generator results for visualization
        splits = list(split_gen)
        self.splits_history[f'Leave-One-{"Group-" if self.groups is not None else ""}Out'] = splits
        
        return iter(splits)

    def stratified_k_fold(self, n_splits=5):
        """
        Performs Stratified K-Fold or Stratified Group K-Fold
        """
        if self.groups is not None:
            print(f"Detected biological groups. Using Stratified Group K-Fold (k={n_splits}).")
            cv = StratifiedGroupKFold(n_splits=n_splits)
            split_gen = cv.split(self.X, self.y, self.groups)
        else:
            print(f"Applying Stratified K-Fold (k={n_splits}).")
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            split_gen = cv.split(self.X, self.y)

        splits = list(split_gen)
        self.splits_history['Stratified K-Fold'] = splits
        return iter(splits)

    def benchmark_methods(self, model=None):
        """
        Benchmarks the different splitting techniques using a simple model
        """
        if model is None:
            # Solver 'liblinear' is good for small datasets, handle numeric inputs only
            model = LogisticRegression(max_iter=1000, solver='liblinear')

        results = []

        # 1. Stratified K-Fold (Group aware)
        print("Benchmarking Stratified K-Fold...")
        k_fold_splits = self.stratified_k_fold(n_splits=5)
        scores_kfold = []
        for train_idx, test_idx in k_fold_splits:
            X_tr, y_tr = self.X.iloc[train_idx], self.y[train_idx]
            X_te, y_te = self.X.iloc[test_idx], self.y[test_idx]
            
            clf = clone(model)
            clf.fit(X_tr, y_tr)
            scores_kfold.append(accuracy_score(y_te, clf.predict(X_te)))
        
        results.append({
            "Method": "Stratified K-Fold",
            "Mean Accuracy": np.mean(scores_kfold),
            "Std Dev": np.std(scores_kfold),
            "Min Score": np.min(scores_kfold),
            "Max Score": np.max(scores_kfold)
        })

        # 2. Leave One Out (Group aware)
        print("Benchmarking LOO/LOGO (This may take time)...")
        loo_splits = self.leave_one_out()
        scores_loo = []
        for train_idx, test_idx in loo_splits:
            X_tr, y_tr = self.X.iloc[train_idx], self.y[train_idx]
            X_te, y_te = self.X.iloc[test_idx], self.y[test_idx]
            
            clf = clone(model)
            clf.fit(X_tr, y_tr)
            scores_loo.append(accuracy_score(y_te, clf.predict(X_te)))

        results.append({
            "Method": "Leave-One-Out (Group-wise)",
            "Mean Accuracy": np.mean(scores_loo),
            "Std Dev": np.std(scores_loo),
            "Min Score": np.min(scores_loo),
            "Max Score": np.max(scores_loo)
        })

        # 3. Simple Split (Repeated)
        print("Benchmarking Repeated 70/30 Split...")
        scores_split = []
        for i in range(10):
            X_tr, X_te, y_tr, y_te = self.split_train_test(test_size=0.3, random_state=i)
            clf = clone(model)
            clf.fit(X_tr, y_tr)
            scores_split.append(accuracy_score(y_te, clf.predict(X_te)))

        results.append({
            "Method": "Repeated Train/Test Split (70/30)",
            "Mean Accuracy": np.mean(scores_split),
            "Std Dev": np.std(scores_split),
            "Min Score": np.min(scores_split),
            "Max Score": np.max(scores_split)
        })

        df_res = pd.DataFrame(results)
        
        # Plotting Benchmark Performance
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Method", y="Mean Accuracy", data=df_res, palette="magma", capsize=.1)
        plt.title("Splitting Technique Benchmark (Model Accuracy)", fontsize=14)
        plt.ylim(0, 1.0)
        plt.ylabel("Accuracy Score")
        plt.tight_layout()
        plt.show()

        return df_res

    def plot_splits_visualization(self):
        """
        Creates a visual graph showing how indices are distributed across Train and Test sets.
        """
        if not self.splits_history:
            print("No splits have been run yet. Run splitting methods before plotting.")
            return

        n_methods = len(self.splits_history)
        fig, axs = plt.subplots(n_methods, 1, figsize=(12, 3 * n_methods), sharex=True)
        
        if n_methods == 1:
            axs = [axs]

        for ax, (method_name, splits) in zip(axs, self.splits_history.items()):
            # Subsample large split sets (like LOO with many samples)
            splits_to_plot = splits[:50] if len(splits) > 50 else splits
            
            n_splits = len(splits_to_plot)
            n_samples = len(self.X)
            
            # Matrix: 0=Unused/Excluded, 1=Training, 2=Testing
            viz_matrix = np.zeros((n_splits, n_samples))

            for i, (train_idx, test_idx) in enumerate(splits_to_plot):
                viz_matrix[i, train_idx] = 1 # Train
                viz_matrix[i, test_idx] = 2  # Test

            # Colors: White (Background), CornflowerBlue (Train), Coral/Orange (Test)
            cmap = plt.cm.colors.ListedColormap(['white', 'cornflowerblue', '#FC8961'])
            ax.imshow(viz_matrix, aspect='auto', cmap=cmap, interpolation='nearest')
            
            ax.set_title(f'{method_name}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Fold/Iter')

        axs[-1].set_xlabel('Sample Index')
        
        # Legend
        legend_elements = [
            Patch(facecolor='cornflowerblue', label='Training Set'),
            Patch(facecolor='#FC8961', label='Test Set')
        ]
        fig.legend(handles=legend_elements, loc='upper right', ncol=2, bbox_to_anchor=(0.5, 1.02))
        
        plt.tight_layout()
        plt.show()

    def _get_positional_indices(self, indices):
        """Helper to convert pandas indices to positional 0..N integers."""
        return [self.X.index.get_loc(idx) for idx in indices]