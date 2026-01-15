import pandas as pd
import numpy as np
from tqdm import tqdm

class DataFusion:
    """
    A class to perform Data Fusion on metabolomic datasets using Multiblock approaches.
    
    This class supports:
    1. Standard Low-Level Data Fusion: Concatenation with block scaling based on total block variance
    2. QC-based Data Fusion: Concatenation with block scaling based on Quality Control (QC) sample variance
    """

    def __init__(self, dataframe_list):
        """
        Initialize the DataFusion object.

        Parameters:
        -----------
        dataframe_list : list of pd.DataFrame
            A list containing the pandas DataFrames to be fused. 
            Each DataFrame represents a data block (e.g., POS/NEG ionization modes).
            Assumes rows are samples and columns are features/metadata.
        """
        self.dataframe_list = dataframe_list
        self.pd_merged = None

    def _process_block(self, df, index, scaling_method='qc'):
        """
        Helper method to process a single data block: separate metadata, 
        calculate scaling factor, and scale features.

        Parameters:
        -----------
        df : pd.DataFrame
            The data block to process.
        index : int
            The index of the block (0-based).
        scaling_method : str
            'qc' for QC-based scaling, 'total' for standard total-block scaling.
        
        Returns:
        --------
        (pd.DataFrame, pd.DataFrame)
            Tuple containing (metadata, scaled_features).
            Metadata is None if index > 0.
        """
        # --- Separate Metadata and Features ---
        # Identify columns with object type (strings) as metadata and numeric as features.
        obs_cols = df.select_dtypes(include=['object']).columns
        num_cols = df.select_dtypes(include=['number']).columns
        
        # Extract metadata only for the first block to avoid duplication
        metadata = None
        if index == 0:
            metadata = df[obs_cols].copy()
        
        features = df[num_cols].copy()

        # --- Determine Scaling Factor ---
        scale_factor = 1.0
        
        if scaling_method == 'qc':
            # QC Identification: Look for "QC" in metadata
            qc_mask = df[obs_cols].apply(lambda x: x.astype(str).str.contains('QC', case=False, na=False)).any(axis=1)
            
            if not qc_mask.any():
                print(f"[WARNING] Block {index+1}: No QC samples found. Defaulting to total block scaling.")
                qc_features = features
            else:
                qc_features = features.loc[qc_mask]
            
            # Calculate Frobenius Norm of QC subset
            norm = np.linalg.norm(qc_features.values, 'fro')
            
        elif scaling_method == 'total':
            # Standard scaling: use the entire block
            norm = np.linalg.norm(features.values, 'fro')
        
        else:
            raise ValueError("Invalid scaling method. Choose 'qc' or 'total'.")

        # Compute scaling factor (inverse of norm)
        if norm == 0:
            print(f"[WARNING] Block {index+1}: Norm is zero. Skipping scaling.")
            scale_factor = 1.0
        else:
            scale_factor = 1.0 / norm

        # --- Apply Scaling ---
        features_scaled = features * scale_factor
        
        # Rename columns to ensure uniqueness (e.g., Feature_Block1)
        features_scaled.columns = [f"{col}_Block{index+1}" for col in features_scaled.columns]

        return metadata, features_scaled

    def low_level_fusion(self):
        """
        Performs Standard Low-Level Data Fusion.

        Strategy:
        Blocks are scaled by the inverse of the Frobenius norm of the *entire* block
        (all samples) before concatenation. This equalizes the variance contribution 
        of each block to the fused dataset.

        Returns:
        --------
        pd.DataFrame
            The fused DataFrame containing metadata and scaled features.
        """
        if not self.dataframe_list or len(self.dataframe_list) < 2:
            print("[ERROR] Fusion requires at least 2 DataFrames")
            return None

        print(f"Starting Standard Low-Level Fusion on {len(self.dataframe_list)} blocks...")
        
        scaled_blocks = []
        final_metadata = None

        for i, df in tqdm(enumerate(self.dataframe_list), total=len(self.dataframe_list), desc="Processing Blocks (Standard)"):
            meta, scaled_feat = self._process_block(df, i, scaling_method='total')
            if i == 0:
                final_metadata = meta
            scaled_blocks.append(scaled_feat)

        # Concatenate
        all_data = [final_metadata] + scaled_blocks
        self.pd_merged = pd.concat(all_data, axis=1)
        
        return self.pd_merged

    def mid_level_fusion(self, n_components=5):
        """
        Performs Mid-Level Data Fusion (Feature Level).

        Strategy:
        1. Separate features from metadata for each block.
        2. Scale features (StandardScaler) within the block.
        3. Perform PCA on each block to extract Principal Components (Scores).
        4. Concatenate the PCA scores from all blocks.

        Parameters:
        -----------
        n_components : int or float
            Number of components to keep per block. 
            If float < 1.0, it selects the number of components explaining that variance fraction.

        Returns:
        --------
        pd.DataFrame
            The fused DataFrame containing metadata and PCA scores from all blocks.
        """
        print(f"Starting Mid-Level Fusion (PCA Scores) on {len(self.dataframe_list)} blocks...")

        score_blocks = []
        final_metadata = None

        for i, df in tqdm(enumerate(self.dataframe_list), total=len(self.dataframe_list), desc="Processing Blocks (Mid-Level)"):
            # 1. Separate Metadata/Features
            obs_cols = df.select_dtypes(include=['object', 'category']).columns
            num_cols = df.select_dtypes(include=['number']).columns
            
            if i == 0:
                final_metadata = df[obs_cols].copy()
            
            features = df[num_cols]

            # 2. Scale (Standardization is crucial for PCA)
            scaler = StandardScaler()
            features_std = scaler.fit_transform(features)

            # 3. PCA Extraction
            pca = PCA(n_components=n_components)
            scores = pca.fit_transform(features_std)
            
            # Create DataFrame for scores
            score_cols = [f"PC{pc+1}_Block{i+1}" for pc in range(scores.shape[1])]
            df_scores = pd.DataFrame(scores, columns=score_cols, index=df.index)
            
            # Print explained variance for info
            expl_var = np.sum(pca.explained_variance_ratio_) * 100
            # tqdm.write(f"  Block {i+1}: Retained {scores.shape[1]} PCs ({expl_var:.1f}% Variance)")

            score_blocks.append(df_scores)

        # 4. Concatenate
        all_data = [final_metadata] + score_blocks
        self.pd_merged = pd.concat(all_data, axis=1)
        
        return self.pd_merged

    def high_level_fusion(self, y, classifier=None):
        """
        Performs High-Level Data Fusion (Decision Level).

        Strategy:
        1. Train a classifier on each block independently.
        2. Predict probabilities (or votes) for each sample from each block.
        3. Concatenate these probabilities to form the fused dataset (or average them).

        Parameters:
        -----------
        y : array-like
            Target labels for the samples. Required for training local models.
        classifier : sklearn estimator, optional
            The model to use for each block. Defaults to RandomForestClassifier.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing class probabilities from each block.
        """
        if classifier is None:
            classifier = RandomForestClassifier(n_estimators=100, random_state=42)

        print(f"Starting High-Level Fusion (Decision Voting) on {len(self.dataframe_list)} blocks...")

        proba_blocks = []
        final_metadata = None

        for i, df in tqdm(enumerate(self.dataframe_list), total=len(self.dataframe_list), desc="Processing Blocks (High-Level)"):
            
            # Separate features
            obs_cols = df.select_dtypes(include=['object', 'category']).columns
            num_cols = df.select_dtypes(include=['number']).columns
            
            if i == 0:
                final_metadata = df[obs_cols].copy()
            
            X = df[num_cols].values
            
            # Generate cross-validated predictions to avoid overfitting on the training set
            # This simulates "unseen" data behavior for the fusion step
            try:
                # predict_proba returns matrix (n_samples, n_classes)
                # We take column [1] for binary classification, or all for multiclass
                probas = cross_val_predict(classifier, X, y, cv=5, method='predict_proba')
                
                # Create column names based on classes
                classes = np.unique(y)
                col_names = [f"Prob_Class{c}_Block{i+1}" for c in classes]
                
                df_proba = pd.DataFrame(probas, columns=col_names, index=df.index)
                proba_blocks.append(df_proba)
                
            except Exception as e:
                print(f"[ERROR] Block {i+1} failed modeling: {e}")

        # Concatenate Probabilities
        all_data = [final_metadata] + proba_blocks
        self.pd_merged = pd.concat(all_data, axis=1)
        
        # Optional: Add a 'Consensus' column (Average of all blocks)
        # Note: This assumes binary classification logic for simplicity in this snippet
        #numeric_cols = self.pd_merged.select_dtypes(include='number')
        #self.pd_merged['Average_Probability'] = numeric_cols.mean(axis=1)

        return self.pd_merged


    def qc_based_fusion(self):
        """
        Performs QC-based Data Fusion.

        Strategy:
        Blocks are scaled by the inverse of the Frobenius norm of the *QC samples only*.
        This normalizes the technical variability (energy) of the analytical platforms 
        based on the reference QC samples, preserving biological variance differences 
        in the study samples.

        Returns:
        --------
        pd.DataFrame
            The fused DataFrame containing metadata and QC-scaled features.
        """
        if not self.dataframe_list or len(self.dataframe_list) < 2:
            print("[ERROR] Fusion requires at least 2 DataFrames.")
            return None

        print(f"Starting QC-based Fusion on {len(self.dataframe_list)} blocks...")
        
        scaled_blocks = []
        final_metadata = None

        for i, df in tqdm(enumerate(self.dataframe_list), total=len(self.dataframe_list), desc="Processing Blocks (QC-based)"):
            meta, scaled_feat = self._process_block(df, i, scaling_method='qc')
            if i == 0:
                final_metadata = meta
            scaled_blocks.append(scaled_feat)

        # Concatenate
        all_data = [final_metadata] + scaled_blocks
        self.pd_merged = pd.concat(all_data, axis=1)
        
        return self.pd_merged