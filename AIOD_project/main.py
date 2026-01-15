import os
import pandas as pd

import src.analysis
from src.analysis import *
from src.anomalyDetection import *
from src.data_loader import *
from src.dataFusion import *
from src.datasetSplitting import *
from src.evaluation import *
from src.models import *
from src.pca import *
from src.plsDa import *
from src.preprocessing import *
from src.simca import *
from src.univariate_analysis import *
from src.visualization import *
from src.datasetSplitting import *

#PATHS CONFIGURATION
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

# --- INPUT PATHS ---
DATA_PATH = os.path.join(PROJECT_PATH, "data")

RAW_DIR = os.path.join(DATA_PATH, "raw")
NEG_RAW_PATH = os.path.join(RAW_DIR, '2024_Metabolomica_Neg.csv')
POS_RAW_PATH = os.path.join(RAW_DIR, '2024_Metabolomica_Pos.csv')

TRANSPOSED_DIR = os.path.join(DATA_PATH, "transposed")
NEG_TRANSPOSED_PATH = os.path.join(TRANSPOSED_DIR, '2024_Metabolomica_Neg_Transposed.csv')
POS_TRANSPOSED_PATH = os.path.join(TRANSPOSED_DIR, '2024_Metabolomica_Pos_Transposed.csv')

CLEANED_DIR = os.path.join(DATA_PATH, "cleaned")
NEG_CLEANED_PATH = os.path.join(CLEANED_DIR, '2024_Metabolomica_Neg_Cleaned.csv')
POS_CLEANED_PATH = os.path.join(CLEANED_DIR, '2024_Metabolomica_Pos_Cleaned.csv')

# --- OUTPUT PATHS (STEP-BASED STRUCTURE) ---
OUT_PATH = os.path.join(PROJECT_PATH, "out")

# STEP 1 ROOT
STEP1_DIR = os.path.join(OUT_PATH, "step_1_preliminary")

# STEP 2 ROOT
STEP2_DIR = os.path.join(OUT_PATH, "step_2_preprocessing")

def main():

    df_neg_raw = load_transposed_dataset(NEG_TRANSPOSED_PATH)
    df_pos_raw = load_transposed_dataset(POS_TRANSPOSED_PATH)
    df_neg = load_transposed_dataset(NEG_CLEANED_PATH)
    df_pos = load_transposed_dataset(POS_CLEANED_PATH)

    os.makedirs(OUT_PATH, exist_ok=True)
    os.makedirs(STEP1_DIR, exist_ok=True)
    os.makedirs(STEP2_DIR, exist_ok=True)

    '''
    #ESECUZIONE COMPLETA PCA
    
    PCA_DIR = os.path.join(STEP2_DIR, "pca")
    PCA_DIR_NEG = os.path.join(PCA_DIR, "neg")
    PCA_DIR_POS = os.path.join(PCA_DIR, "pos")

    results_neg = perform_pca(df_neg, scaling="autoscaling")
    plot_pca_scores(results_neg, df_neg, output_dir=PCA_DIR_NEG, show_ellipse= False)
    plot_pca_loadings(results_neg, output_dir=PCA_DIR_NEG, top_n= 10)
    plot_pca_scree(results_neg, output_dir=PCA_DIR_NEG, threshold= 0.95)
    plot_loading_profile(results_neg, output_dir=PCA_DIR_NEG, pc_index= 1, top_n= 10)
    
    results_pos = perform_pca(df_pos, scaling="autoscaling")
    plot_pca_scores(results_pos, df_pos, output_dir=PCA_DIR_POS, show_ellipse= False)
    plot_pca_loadings(results_pos, output_dir=PCA_DIR_POS, top_n= 10)
    plot_pca_scree(results_pos, output_dir=PCA_DIR_POS, threshold= 0.95)
    plot_loading_profile(results_pos, output_dir=PCA_DIR_POS, pc_index= 1, top_n= 10)
    '''

    PCA_DIR_NO_NORM = os.path.join(STEP2_DIR, "pca_without_normalization")
    PCA_DIR_NEG_NO_NORM = os.path.join(PCA_DIR_NO_NORM, "neg")
    PCA_DIR_POS_NO_NORM = os.path.join(PCA_DIR_NO_NORM, "pos")

    results_neg = perform_pca(df_neg, scaling="autoscaling")
    plot_pca_scores(results_neg, df_neg, output_dir=PCA_DIR_NEG_NO_NORM, show_ellipse=False)
    plot_pca_loadings(results_neg, output_dir=PCA_DIR_NEG_NO_NORM, top_n=10)
    plot_pca_scree(results_neg, output_dir=PCA_DIR_NEG_NO_NORM, threshold=0.95)
    plot_loading_profile(results_neg, output_dir=PCA_DIR_NEG_NO_NORM, pc_index=1, top_n=10)

    results_pos = perform_pca(df_pos, scaling="autoscaling")
    plot_pca_scores(results_pos, df_pos, output_dir=PCA_DIR_POS_NO_NORM, show_ellipse=False)
    plot_pca_loadings(results_pos, output_dir=PCA_DIR_POS_NO_NORM, top_n=10)
    plot_pca_scree(results_pos, output_dir=PCA_DIR_POS_NO_NORM, threshold=0.95)
    plot_loading_profile(results_pos, output_dir=PCA_DIR_POS_NO_NORM, pc_index=1, top_n=10)

    #NORMALIZATION
    df_neg = normalization_pqn(df_neg)
    df_pos = normalization_pqn(df_pos)

    PCA_DIR_NORM = os.path.join(STEP2_DIR, "pca_with_normalization")
    PCA_DIR_NEG_NORM = os.path.join(PCA_DIR_NORM, "neg")
    PCA_DIR_POS_NORM = os.path.join(PCA_DIR_NORM, "pos")

    results_neg = perform_pca(df_neg, scaling="autoscaling")
    plot_pca_scores(results_neg, df_neg, output_dir=PCA_DIR_NEG_NORM, show_ellipse=False)
    plot_pca_loadings(results_neg, output_dir=PCA_DIR_NEG_NORM, top_n=10)
    plot_pca_scree(results_neg, output_dir=PCA_DIR_NEG_NORM, threshold=0.95)
    plot_loading_profile(results_neg, output_dir=PCA_DIR_NEG_NORM, pc_index=1, top_n=10)

    results_pos = perform_pca(df_pos, scaling="autoscaling")
    plot_pca_scores(results_pos, df_pos, output_dir=PCA_DIR_POS_NORM, show_ellipse=False)
    plot_pca_loadings(results_pos, output_dir=PCA_DIR_POS_NORM, top_n=10)
    plot_pca_scree(results_pos, output_dir=PCA_DIR_POS_NORM, threshold=0.95)
    plot_loading_profile(results_pos, output_dir=PCA_DIR_POS_NORM, pc_index=1, top_n=10)

    # ==============================================================================
#  DATASET MERGING
# ==============================================================================

    df_list = []

    df_list.append(df_neg_clean)
    df_list.append(df_pos_clean)
    """
    dFusion = DataFusion(df_list)
    df_low_level_merged = dFusion.low_level_fusion()
    #df_qc_merged = dFusion.qc_based_fusion()

    #print(df_low_level_merged)
    #print(df_qc_merged)

    
    plots_dir = os.path.join(STEP2_DIR, "plots")

    
    biplot(df_neg_clean, "Negative cleaned", plots_dir)
    biplot(df_pos_clean, "Positive cleaned", plots_dir)

    sumPCA(df_low_level_merged, "Low Level Fusion", plots_dir)


    z_score_plot(df_neg_clean, "Z-score negative cleaned", plots_dir)


    internal_variability(df_neg_clean, "Internal variability negative cleaned", plots_dir)
    """

# ==============================================================================
#  DATASET SPLITTING
# ==============================================================================

    splitting_plots_dir = os.path.join(STEP2_DIR, "plots splitting")

    #print(df_neg_clean.columns)
    splitter = DatasetSplitting(df_neg_clean, target_col="Class")

    # Train/Test Split
    # Returns: Tuple (X_train, X_test, y_train, y_test)
    df_train_test_split = splitter.split_train_test(test_size=0.25, normalize=True)
    print(f"\nTrain/Test Split\nTrain shape {df_train_test_split[0].shape}, Test shape {df_train_test_split[1].shape}")

    # Leave-One-Group-Out (LOGO)
    # Returns: List of tuples [(X_train, X_test, y_train, y_test), ...]
    df_loo_split = splitter.leave_one_out(normalize=True)
    print(f"\nLOO Split\nCreated {len(df_loo_split)} folds.")

    # Stratified Group K-Fold
    # Returns: List of tuples
    df_stratified_k_fold = splitter.stratified_k_fold(n_splits=4, normalize=True)
    print(f"\nStratified K-Fold\nCreated {len(df_stratified_k_fold)} folds.")

    # Kennard-Stone (Rational Split)
    # Returns: List containing one tuple [(X_train, X_test, y_train, y_test)]
    df_kennard_stone_split = splitter.kennard_stone_split(train_size=0.75, normalize=True)
    print(f"\nKennard-Stone\nTrain shape {df_kennard_stone_split[0][0].shape}")

    # Duplex (Rational Split)
    # Returns: List containing one tuple
    df_duplex_split = splitter.duplex_split(split_ratio=0.75, normalize=True)
    print(f"\nDuplex\nTrain shape {df_duplex_split[0][0].shape}")

    # Onion (Distance Sorted Split)
    # Returns: List containing one tuple
    df_onion_split = splitter.onion_split(test_size=0.25, normalize=True)
    print(f"\nOnion\nTrain shape {df_onion_split[0][0].shape}")


    """
    splitter.plot_splits(df_train_test_split, "Train-Test Split split", splitting_plots_dir)
    splitter.plot_splits(df_loo_split, "Leave-One-Group-Out split", splitting_plots_dir, fold_index=0)
    splitter.plot_splits(df_stratified_k_fold, "K-fold split", splitting_plots_dir, fold_index=0)
    splitter.plot_splits(df_kennard_stone_split, "Kennard split", splitting_plots_dir)
    splitter.plot_splits(df_duplex_split, "Duplex split", splitting_plots_dir)
    splitter.plot_splits(df_onion_split, "Onion split", splitting_plots_dir)
    """

    #splitter.plot_feature_scatter(df_loo_split, "Leave-One-Group-Out split plot2", splitting_plots_dir, feature_indices=(0, 1))

# ==============================================================================
#  ANOMALIES DETECTION
# ==============================================================================

    detector = AnomalyDetector(df_neg_clean) 

    # 2. Visualizing Mahalanobis Contours
    # This method creates a specific plot for Robust Covariance
    print("\n--- 1. Mahalanobis Distance Analysis ---")
    detector.plot_mahalanobis_contours(df_neg_clean, title="Negative Dataset")
    # 3. Visualizing Algorithm Grid
    # This method compares multiple algorithms on a 2D grid.
    # We must project the data to 2D (PCA) before passing it to the grid plotter.
    print("\n--- 2. Algorithm Comparison Grid ---")
    X_pca_2d, _ = detector._generate_pca(df_neg_clean, n_components=2)
    detector.plot_anomaly_grid(
        X_pca_2d, 
        detector.algorithms, 
        fname="Negative_Dataset_Anomaly_Grid", 
        save_plots=False
    )
    # 4. Z-Score Analysis
    # Calculates outliers based on total sample intensity (Total Ion Current proxy)
    print("\n--- 3. Z-Score Analysis ---")
    z_scores, std_dev = detector.calculate_z_scores(df_neg_clean)

    # Print summary
    print(f"Z-Score Stats -> Mean: {z_scores.mean():.4f}, Std Dev (Intensity): {std_dev:.4f}")

    # Identify and print potential outliers (e.g., > 3 sigma)
    outliers_z = z_scores[np.abs(z_scores) > 3]
    if not outliers_z.empty:
        print(f"Potential Intensity Outliers (>3 sigma): {len(outliers_z)}")
        print(outliers_z.index.tolist())
    else:
        print("No extreme intensity outliers (>3 sigma) detected.")
    # 5. Benchmarking
    # Runs all algorithms on the full dataset and scores them using Silhouette Score
    print("\n--- 4. Benchmarking Algorithms ---")
    benchmark_results = detector.benchmark_algorithms(df_neg_clean, fname="Negative Dataset")
    # The benchmark method prints the table automatically, but it is also returned here




if __name__ == "__main__":
    '''
    os.makedirs(TRANSPOSED_DIR, exist_ok=True)
    reshape_dataset(load_raw_dataset(NEG_RAW_PATH)).to_csv(NEG_TRANSPOSED_PATH, index=True)
    reshape_dataset(load_raw_dataset(POS_RAW_PATH)).to_csv(POS_TRANSPOSED_PATH, index=True)

    os.makedirs(CLEANED_DIR, exist_ok=True)
    remove_qc_and_technical_replicates(load_transposed_dataset(NEG_TRANSPOSED_PATH)).to_csv(NEG_CLEANED_PATH, index=True)
    remove_qc_and_technical_replicates(load_transposed_dataset(POS_TRANSPOSED_PATH)).to_csv(POS_CLEANED_PATH, index=True)
    '''
    main()