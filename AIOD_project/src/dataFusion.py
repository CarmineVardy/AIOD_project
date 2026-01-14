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