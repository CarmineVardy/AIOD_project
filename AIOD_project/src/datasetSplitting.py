import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    LeaveOneOut,
    GroupShuffleSplit,
    StratifiedGroupKFold,
    LeaveOneGroupOut
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.visualization import save_plot


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


    """
    Normalizes the features (X) to scale data (e.g., Mean=0, Std=1)
    
    (?)To prevent Data Leakage, the scaler is FIT only on X_train, 
    and then that fitted scaler is used to TRANSFORM X_test.
    
    Arguments:
        input_data: Can be two types:
            1. A list of tuples: [(X_train, X_test, y_train, y_test), ...]
            2. A single tuple: (X_train, X_test, y_train, y_test)
        method (str): 'standard' (Z-score) or 'minmax' (0-1 scaling).
        
    This function returns the same structure as input_data (List or Tuple), but with X normalized
    """
    def normalize_split(self, input_data, method='standard', normalize=False):
        
        # Select the Scaler
        if method == 'minmax':
            scaler_cls = MinMaxScaler
        else:
            scaler_cls = StandardScaler # Default
            
        # --- Internal Helper to normalize one single Train/Test pair ---
        def _apply_norm(X_tr, X_te, y_tr, y_te):
            # Create a fresh scaler for this specific fold/split
            scaler = scaler_cls()
            
            # 1. Fit on TRAIN, Transform TRAIN
            # Check if input is a Pandas DataFrame to preserve columns/indices
            if hasattr(X_tr, "columns"): 
                X_tr_scaled = pd.DataFrame(
                    scaler.fit_transform(X_tr), 
                    index=X_tr.index, 
                    columns=X_tr.columns
                )
                
                # 2. Transform TEST (using the stats from Train)
                X_te_scaled = pd.DataFrame(
                    scaler.transform(X_te), 
                    index=X_te.index, 
                    columns=X_te.columns
                )
            else:
                # Numpy array handling
                X_tr_scaled = scaler.fit_transform(X_tr)
                X_te_scaled = scaler.transform(X_te)
                
            return (X_tr_scaled, X_te_scaled, y_tr, y_te)

        # --- Main Logic to handle List vs Tuple input ---
        
        # Case A: Input is a List of folds (e.g., from K-Fold or LOO)
        if isinstance(input_data, list):
            normalized_list = []
            #print(f"Normalizing {len(input_data)} splits using {method} scaler...")
            for split in input_data:
                # Unpack, normalize, repack
                normalized_list.append(_apply_norm(*split))
            return normalized_list
            
        # Case B: Input is a Single Tuple (e.g., from simple split, KS, Duplex)
        elif isinstance(input_data, tuple) and len(input_data) == 4:
            #print(f"Normalizing single split using {method} scaler...")
            return _apply_norm(*input_data)
            
        else:
            raise ValueError("Input must be either a list of tuples or a single (X_train, X_test, y_train, y_test) tuple")


    """
    Performs a simple Train/Test split 75/25
    Resects biological groups if 'group_col' is provided
    
    Group-Aware Logic: If groups are present, it uses GroupShuffleSplit. This ensures that all samples belonging to a specific group 
                        (e.g., all replicates from Patient A) are assigned exclusively to either the training set or the test set. 
                        This prevents the model from "memorizing" a patient's specific profile.

    Standard Logic: If no groups are present, it performs a standard train_test_split with stratification 
                    (maintaining the proportion of Control/Case classes)
    
    This method returns a DataFrames/Arrays directly
    """

    def split_train_test(self, test_size=0.25, random_state=42, normalize=False):
        
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
        
        '''
        X_train, X_test -> set divided
        y_train, y_test -> labels
        '''

        if normalize:
            # Passing the single tuple to the normalization function
            X_train, X_test, y_train, y_test = self.normalize_split(
                (X_train, X_test, y_train, y_test), method='standard'
            )

        return X_train, X_test, y_train, y_test



    """
    Leave-One-Group-Out (LOGO): If biological groups are detected, it automatically switches from standard LOO to LOGO. 
                                Instead of leaving out a single sample, it leaves out an entire biological group 
                                (e.g., "Leave One Patient Out"). This is the correct validation approach for biological studies 
                                with replicates.

    Standard LOO: If no groups exist, it proceeds with standard Leave-One-Out, identifying a single sample as the test set in each iteration.
    
    This methods return a list of tuples, where each tuple contains (X_train, X_test, y_train, y_test)
    """
    def leave_one_out(self, normalize=False):
        # Prepare to store training and test sets
        train_test_sets = []

        if self.groups is not None:
            print(f"Detected biological groups. Switching LOO to Leave-One-Group-Out ({self.group_col})")
            cv = LeaveOneGroupOut()
            split_gen = cv.split(self.X, self.y, self.groups)
        else:
            print("Applying standard Leave-One-Out.")
            cv = LeaveOneOut()
            split_gen = cv.split(self.X, self.y)
        
        # Generate splits and create corresponding train/test sets
        for train_indices, test_indices in list(split_gen):
            # --- FIX: Use .iloc for Pandas DataFrame row slicing ---
            if hasattr(self.X, "iloc"):
                X_train = self.X.iloc[train_indices]
                X_test = self.X.iloc[test_indices]
            else:
                # Fallback if self.X is a numpy array
                X_train = self.X[train_indices]
                X_test = self.X[test_indices]
                
            # self.y is already a numpy array (from __init__), so simple indexing works
            y_train = self.y[train_indices]
            y_test = self.y[test_indices]
            
            # Store the sets of this split
            train_test_sets.append((X_train, X_test, y_train, y_test))
        
        if normalize:
            # Passing the whole list to the normalization function
            train_test_sets = self.normalize_split(train_test_sets, method='standard')

        # Store split history
        self.splits_history[f'Leave-One-{"Group-" if self.groups is not None else ""}Out'] = train_test_sets
        
        return train_test_sets



    """
    This method splits the data into k subsets (folds)

    Stratified Group K-Fold: If groups are present, it uses StratifiedGroupKFold. This is a complex splitter that attempts to satisfy 
                            two constraints simultaneously:
        - Independence: Groups are not split across folds
        - Stratification: The ratio of classes (Control/Case) is preserved as much as possible in each fold

    Standard Stratified K-Fold: Without groups, it simply ensures class balance across the random folds.

    Returns a list of (X_train, X_test, y_train, y_test) tuples
    """
    def stratified_k_fold(self, n_splits=5, normalize=False):
        train_test_sets = []

        # 1. Choose the correct Cross-Validator based on groups
        if self.groups is not None:
            print(f"Detected biological groups. Using Stratified Group K-Fold (k={n_splits})")
            cv = StratifiedGroupKFold(n_splits=n_splits)
            split_gen = cv.split(self.X, self.y, self.groups)
        else:
            print(f"Applying Stratified K-Fold (k={n_splits}).")
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            split_gen = cv.split(self.X, self.y)

        # 2. Iterate through indices and slice the actual data
        for train_indices, test_indices in split_gen:
            
            # --- FIX: Use .iloc for Pandas DataFrame row slicing ---
            if hasattr(self.X, "iloc"):
                X_train = self.X.iloc[train_indices]
                X_test = self.X.iloc[test_indices]
            else:
                X_train = self.X[train_indices]
                X_test = self.X[test_indices]
            
            y_train = self.y[train_indices]
            y_test = self.y[test_indices]                      

            train_test_sets.append((X_train, X_test, y_train, y_test))

        if normalize:
            # Passing the whole list to the normalization function
            train_test_sets = self.normalize_split(train_test_sets, method='standard')

        # 3. Store history and return
        self.splits_history['Stratified K-Fold'] = train_test_sets
        return train_test_sets


    """
    Kennard-Stone (KS) Algorithm

    A deterministic method that selects samples to uniformly cover the predictor space (X).
    1. Selects the two samples with the largest Euclidean distance.
    2. Iteratively selects the sample with the largest minimum distance to the samples already selected.

    Returns:
        List containing one tuple: [(X_train, X_test, y_train, y_test)]

    ⚠ For very large dataset, Fast Kennard-Stone is better to avoid computing the full matrix at once
    """
    def kennard_stone_split(self, train_size=0.75, normalize=False):
        print(f"Applying Kennard-Stone Split (Train ratio={train_size})...")

        # 1. Prepare Data
        X_data = self.X.values if hasattr(self.X, "values") else self.X
        n_samples = X_data.shape[0]
        n_train = int(n_samples * train_size)

        # Indices list
        remaining_indices = list(range(n_samples))
        selected_indices = []

        # 2. Initialization: Find 2 most distant points
        dist_matrix = cdist(X_data, X_data, metric='euclidean')
        i, j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
        selected_indices.extend([i, j])
        remaining_indices.remove(i)
        remaining_indices.remove(j)

        # 3. Iterative Selection (Maximin)
        for _ in range(n_train - 2):
            dist_to_selected = dist_matrix[np.ix_(remaining_indices, selected_indices)]
            min_dists = np.min(dist_to_selected, axis=1)
            best_candidate_idx = np.argmax(min_dists)
            
            actual_idx = remaining_indices[best_candidate_idx]
            selected_indices.append(actual_idx)
            remaining_indices.pop(best_candidate_idx)

        # 4. Construct Sets
        train_indices = np.array(selected_indices)
        test_indices = np.array(remaining_indices)

        # --- FIX: Only use .iloc for self.X (if it's a DataFrame) ---
        if hasattr(self.X, "iloc"):
            X_train, X_test = self.X.iloc[train_indices], self.X.iloc[test_indices]
        else:
            X_train, X_test = self.X[train_indices], self.X[test_indices]
            
        # self.y is a numpy array, so always use brackets []
        y_train, y_test = self.y[train_indices], self.y[test_indices]

        if normalize:
            X_train, X_test, y_train, y_test = self.normalize_split(
                (X_train, X_test, y_train, y_test), method='standard'
            )

        train_test_sets = [(X_train, X_test, y_train, y_test)]
        self.splits_history['Kennard-Stone'] = train_test_sets

        return train_test_sets



    """
    Duplex Algorithm

    Similar to Kennard-Stone but splits data into two sets (Train and Test) simultaneously 
    to ensure both cover the space equally well.
    1. Select two furthest points -> Train.
    2. Select two furthest points from remaining -> Test.
    3. Repeat alternating selection.

    Returns:
        List containing one tuple: [(X_train, X_test, y_train, y_test)]
    """
    def duplex_split(self, split_ratio=0.75, normalize=False):
        print(f"Applying Duplex Split (Split ratio={split_ratio})...")

        X_data = self.X.values if hasattr(self.X, "values") else self.X
        n_samples = X_data.shape[0]
        n_train = int(n_samples * split_ratio)

        remaining_indices = list(range(n_samples))
        train_indices = []
        test_indices = []
        dist_matrix = cdist(X_data, X_data, metric='euclidean')

        while len(remaining_indices) > 0:
            # 1. Select for TRAIN
            if len(train_indices) < n_train:
                if len(train_indices) == 0:
                    sub_dist = dist_matrix[np.ix_(remaining_indices, remaining_indices)]
                    i, j = np.unravel_index(np.argmax(sub_dist), sub_dist.shape)
                    real_i, real_j = remaining_indices[i], remaining_indices[j]
                    train_indices.extend([real_i, real_j])
                    remaining_indices.remove(real_i)
                    if real_j in remaining_indices: remaining_indices.remove(real_j)
                else:
                    dist_to_train = dist_matrix[np.ix_(remaining_indices, train_indices)]
                    min_dists = np.min(dist_to_train, axis=1)
                    best_idx = np.argmax(min_dists)
                    train_indices.append(remaining_indices.pop(best_idx))

            # 2. Select for TEST
            if len(remaining_indices) > 0:
                if len(test_indices) == 0:
                    sub_dist = dist_matrix[np.ix_(remaining_indices, remaining_indices)]
                    i, j = np.unravel_index(np.argmax(sub_dist), sub_dist.shape)
                    real_i, real_j = remaining_indices[i], remaining_indices[j]
                    test_indices.extend([real_i, real_j])
                    remaining_indices.remove(real_i)
                    if real_j in remaining_indices: remaining_indices.remove(real_j)
                else:
                    dist_to_test = dist_matrix[np.ix_(remaining_indices, test_indices)]
                    min_dists = np.min(dist_to_test, axis=1)
                    best_idx = np.argmax(min_dists)
                    test_indices.append(remaining_indices.pop(best_idx))

        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)

        # --- FIX: Handle slicing properly ---
        if hasattr(self.X, "iloc"):
            X_train, X_test = self.X.iloc[train_indices], self.X.iloc[test_indices]
        else:
            X_train, X_test = self.X[train_indices], self.X[test_indices]
            
        y_train, y_test = self.y[train_indices], self.y[test_indices]

        if normalize:
            X_train, X_test, y_train, y_test = self.normalize_split(
                (X_train, X_test, y_train, y_test), method='standard'
            )

        train_test_sets = [(X_train, X_test, y_train, y_test)]
        self.splits_history['Duplex'] = train_test_sets

        return train_test_sets



    """
    Onion (Sorted Distance) Split

    Stratifies samples based on their distance to the centroid (Mean of X).
    Data is effectively treated as layers of an "onion".
    1. Calculate centroid.
    2. Sort all samples by distance to centroid.
    3. Perform systematic sampling (e.g., take every Nth sample for test) to ensure
       the test set contains samples from the center, the middle layers, and the edges.

    Returns:
        List containing one tuple: [(X_train, X_test, y_train, y_test)]
    """
    def onion_split(self, test_size=0.25, normalize=False):
        print(f"Applying Onion (Distance-Sorted) Split (Test size={test_size})...")

        X_data = self.X.values if hasattr(self.X, "values") else self.X
        n_samples = X_data.shape[0]

        centroid = np.mean(X_data, axis=0).reshape(1, -1)
        dists = cdist(X_data, centroid, metric='euclidean').flatten()
        sorted_indices = np.argsort(dists)

        step = int(1 / test_size)
        
        # Systematic sampling on sorted indices
        test_selection = sorted_indices[1::step]
        train_selection = np.setdiff1d(sorted_indices, test_selection)

        # --- FIX: Handle slicing properly ---
        if hasattr(self.X, "iloc"):
            X_train, X_test = self.X.iloc[train_selection], self.X.iloc[test_selection]
        else:
            X_train, X_test = self.X[train_selection], self.X[test_selection]
            
        y_train, y_test = self.y[train_selection], self.y[test_selection]
        
        if normalize:
            X_train, X_test, y_train, y_test = self.normalize_split(
                (X_train, X_test, y_train, y_test), method='standard'
            )

        train_test_sets = [(X_train, X_test, y_train, y_test)]
        self.splits_history['Onion'] = train_test_sets

        return train_test_sets



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

    """
    Visualizes the Train/Test split in 2D space using PCA.
    It combines X_train and X_test, runs PCA, and plots them with different colors.
    
    Arguments:
        train_test_sets: The output from any splitting function (List of tuples).
                         Note: If multiple folds exist (e.g., K-Fold), it plots only the fold 
                         specified by 'fold_index'.
        filename (str): Name of the file to save (e.g., "onion_split.png").
        directory (str): Directory to save the plot.
        fold_index (int): Which fold to visualize (default 0).

    Automatically handles both single splits (tuple) and multiple folds (list).
    """
    def plot_splits(self, train_test_sets, filename, directory, fold_index=0):
        
        #print(f"Generating split visualization for '{filename}'...")

        # 2. Extract the Data (Handle List vs Tuple input)
        
        # Case A: Input is a Single Tuple (from split_train_test) -> (X_tr, X_te, y_tr, y_te)
        # We check if it's a tuple AND the first element is NOT a tuple/list (it should be a DataFrame/Array)
        if isinstance(train_test_sets, tuple) and len(train_test_sets) == 4 and not isinstance(train_test_sets[0], (list, tuple)):
            X_train, X_test, y_train, y_test = train_test_sets
            
        # Case B: Input is a List of folds (from KS, Duplex, Onion, K-Fold) -> [(X,X,y,y), ...]
        elif isinstance(train_test_sets, list):
            if fold_index >= len(train_test_sets):
                print(f"Warning: Fold index {fold_index} out of range. Using fold 0.")
                fold_index = 0
            X_train, X_test, y_train, y_test = train_test_sets[fold_index]
            
        else:
            raise ValueError("Input to plot_splits must be either a (X,X,y,y) tuple or a list of such tuples")

        # 3. Prepare Data for PCA 
        # (We combine Train + Test to fit PCA globally so the projection is consistent)
        
        # Check if inputs are DataFrames or Arrays
        if hasattr(X_train, "values"):
            X_combined = np.vstack((X_train.values, X_test.values))
        else:
            X_combined = np.vstack((X_train, X_test))
            
        # Track counts for legend
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]
        
        # 4. Compute PCA (2 Components)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_combined)
        
        # 5. Plotting
        plt.figure(figsize=(10, 7))
        
        # Plot Train points (First n_train rows)
        plt.scatter(X_pca[:n_train, 0], X_pca[:n_train, 1], 
                    c='royalblue', label=f'Train ({n_train})', alpha=0.7, edgecolors='k', s=80)
        
        # Plot Test points (Remaining rows)
        plt.scatter(X_pca[n_train:, 0], X_pca[n_train:, 1], 
                    c='darkorange', label=f'Test ({n_test})', alpha=0.9, edgecolors='k', marker='D', s=80)
        

        plt.axhline(0, color='#050402', linestyle='solid', linewidth=0.8, alpha=0.8)
        plt.axvline(0, color='#050402', linestyle='solid', linewidth=0.8, alpha=0.8)
        plt.grid(color='gray', linestyle='dashed', linewidth=0.5, alpha=0.7)

        # Add labels and title
        plt.title(f"Split Visualization: {filename.replace('.png', '')}", fontsize=14)
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})", fontsize=12)
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # 6. Save and Close
        save_plot(plt, filename, directory)
        #plt.show()


    """
    Visualizes the Train/Test split by plotting two specific raw features against each other
    (No PCA involved).
    
    Arguments:
        train_test_sets: Input split data (Tuple or List).
        filename (str): Output filename.
        feature_indices (tuple): Indices of the two features to plot (e.g., (0, 1) for first two columns).
        directory (str): Output directory.
        fold_index (int): Fold to visualize.
    """
    def plot_feature_scatter(self, filename, directory, train_test_sets, feature_indices=(0, 1), fold_index=0):
        
        print(f"Generating feature scatter plot for '{filename}'...")

        # 2. Extract Data (Handle Tuple vs List)
        if isinstance(train_test_sets, tuple) and len(train_test_sets) == 4 and not isinstance(train_test_sets[0], (list, tuple)):
            X_train, X_test, y_train, y_test = train_test_sets
        elif isinstance(train_test_sets, list):
            if fold_index >= len(train_test_sets): fold_index = 0
            X_train, X_test, y_train, y_test = train_test_sets[fold_index]
        else:
            raise ValueError("Input must be a (X,X,y,y) tuple or a list of such tuples.")

        # 3. Extract Specific Features
        idx1, idx2 = feature_indices
        
        # Helper to get column data whether it's DataFrame or Numpy
        def get_col(data, col_idx):
            if hasattr(data, "iloc"):
                return data.iloc[:, col_idx].values
            return data[:, col_idx]

        # Get feature names for labels (if DataFrame)
        x_label = f"Feature {idx1}"
        y_label = f"Feature {idx2}"
        if hasattr(X_train, "columns"):
            x_label = X_train.columns[idx1]
            y_label = X_train.columns[idx2]

        # 4. Plotting
        plt.figure(figsize=(10, 7))
        
        # Plot Train
        plt.scatter(get_col(X_train, idx1), get_col(X_train, idx2),
                    c='royalblue', label=f'Train ({len(X_train)})', alpha=0.7, edgecolors='k', s=80)
        
        # Plot Test
        plt.scatter(get_col(X_test, idx1), get_col(X_test, idx2),
                    c='darkorange', label=f'Test ({len(X_test)})', alpha=0.9, edgecolors='k', marker='D', s=80)

        plt.title(f"Raw Feature Split: {x_label} vs {y_label}", fontsize=14)
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

        # 6. Save and Close
        save_plot(plt, filename, directory)
        #plt.show()


    def _get_positional_indices(self, indices):
        """Helper to convert pandas indices to positional 0..N integers."""
        return [self.X.index.get_loc(idx) for idx in indices]