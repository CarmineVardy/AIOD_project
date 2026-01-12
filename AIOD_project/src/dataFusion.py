import pandas as pd
import numpy as np
from tqdm import tqdm

class DataFusion:
    """
    A class to perform Low-Level Data Fusion on metabolomic datasets using a Multiblock approach.
    
    This implementation follows the Low-Level Fusion strategy where raw data blocks are 
    pre-processed (scaled) and then concatenated. To ensure equal contribution from 
    different platforms (or blocks), we apply Block Scaling.
    
    Specific Strategy:
    Instead of scaling by the total block variance (standard Frobenius norm), this class 
    exploits Quality Control (QC) samples. Each block is scaled by the Frobenius norm 
    calculated strictly on its QC subset. This effectively normalizes the 'technical energy' 
    of the blocks before fusion.
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

    def low_level_fusion(self):
        """
        Performs Low-Level Data Fusion (concatenation) with QC-based Block Scaling.

        The process involves:
        1. Validation: Ensures at least 2 blocks are provided.
        2. Feature Extraction: Separates numeric features from metadata.
        3. QC Identification: Automatically detects QC samples based on 'QC' string in metadata.
        4. Block Scaling: Scales the entire block by the inverse of the Frobenius norm of its QC samples.
           Scaling Factor = 1 / || X_QC ||_F
        5. Fusion: Horizontally concatenates the scaled blocks.

        Returns:
        --------
        pd.DataFrame
            The final fused DataFrame stored in self.pd_merged, containing metadata 
            and all scaled features.
        """
        
        # Check if we have enough blocks for fusion
        if not self.dataframe_list or len(self.dataframe_list) < 2:
            print("[ERROR] Fusion requires at least 2 DataFrame")
            return None

        print(f"Initializing Low-Level Data Fusion on {len(self.dataframe_list)} blocks...")
        
        scaled_blocks = []
        metadata = None

        # Iterate through each dataframe with a progress loading bar
        for i, df in tqdm(enumerate(self.dataframe_list), total=len(self.dataframe_list), desc="Processing Blocks"):
            
            # --- Step 1: Separate Metadata and Features ---
            # We identify columns with object type (strings) as metadata and numeric as features.
            obs_cols = df.select_dtypes(include=['object']).columns
            num_cols = df.select_dtypes(include=['number']).columns
            
            # Use the metadata from the first block as the anchor for the final dataframe
            if i == 0:
                metadata = df[obs_cols].copy()
            else:
                # In a robust production environment, we would check sample alignment here.
                # For this implementation, we assume input dataframes are row-aligned.
                pass

            features = df[num_cols].copy()

            # --- Step 2: QC Identification ---
            # We look for the string "QC" in any of the metadata columns (case-insensitive)
            qc_mask = df[obs_cols].apply(lambda x: x.astype(str).str.contains('QC', case=False, na=False)).any(axis=1)
            
            # Fallback mechanism if no QC samples are found
            if not qc_mask.any():
                print(f"[WARNING] No QC samples found in Block {i+1}. Defaulting to standard Frobenius scaling (all samples)")
                qc_features = features
            else:
                qc_features = features.loc[qc_mask]

            # --- Step 3: Compute QC-based Scaling Factor ---
            # Calculate the Frobenius Norm of the QC sub-matrix: sqrt(sum(x_ij^2))
            qc_frobenius_norm = np.linalg.norm(qc_features.values, 'fro')

            # Prevent division by zero
            if qc_frobenius_norm == 0:
                print(f"[WARNING] Block {i+1} QC matrix has zero norm. Skipping scaling for this block.")
                scale_factor = 1.0
            else:
                # The scaling factor ensures the QC 'energy' is normalized to 1 across blocks
                scale_factor = 1.0 / qc_frobenius_norm

            # --- Step 4: Apply Scaling to the Block ---
            # Scale the entire feature set (Samples + QCs) using the factor derived from QCs
            features_scaled = features * scale_factor
            
            # Rename columns to ensure uniqueness and traceability in the fused dataset
            features_scaled.columns = [f"{col}_Block{i+1}" for col in features_scaled.columns]
            
            scaled_blocks.append(features_scaled)

        # --- Step 5: Low-Level Fusion (Concatenation) ---
        
        # Combine metadata and all scaled feature blocks horizontally
        all_data = [metadata] + scaled_blocks
        self.pd_merged = pd.concat(all_data, axis=1)
        
        return self.pd_merged