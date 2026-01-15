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

    plots_dir = os.path.join(STEP2_DIR, "plots")

    #plot_class_distribution_pie(df_neg_raw, plots_dir, "Datset Negative raw")
    #plot_class_distribution_pie(df_pos_raw, plots_dir, "Datset Positive raw")

    biplot(df_neg_raw, "Negative Raw", plots_dir)
    biplot(df_pos_raw, "Positive Raw", plots_dir)

    df_list = []
    df_list.append(df_neg_raw)
    df_list.append(df_pos_raw)
    dFusion = DataFusion(df_list)
    df_low_level_merged = dFusion.low_level_fusion()

    sumPCA(df_low_level_merged, "Low Level Fusion", plots_dir)

    z_score_plot(df_neg_raw, "Z-score negative Raw", plots_dir)
    z_score_plot(df_pos_raw, "Z-score positive Raw", plots_dir)

    internal_variability(df_neg_raw, "Internal variability negative Raw", plots_dir)
    internal_variability(df_pos_raw, "Internal variability positive Raw", plots_dir)


# ==============================================================================
#  ANOMALIES DETECTION
# ==============================================================================

    detector = AnomalyDetector() 

    # 2. Visualizing Mahalanobis Contours
    # This method creates a specific plot for Robust Covariance
    print("\n--- 1. Mahalanobis Distance Analysis ---")
    detector.plot_mahalanobis_contours(df_neg_raw, title="Negative Dataset")
    # 3. Visualizing Algorithm Grid
    # This method compares multiple algorithms on a 2D grid.
    # We must project the data to 2D (PCA) before passing it to the grid plotter.
    print("\n--- 2. Algorithm Comparison Grid ---")
    X_pca_2d, _ = detector._generate_pca(df_neg_raw, n_components=2)
    detector.plot_anomaly_grid(
        X_pca_2d, 
        detector.algorithms, 
        fname="Negative_Dataset_Anomaly_Grid", 
        save_plots=False
    )
    # 4. Z-Score Analysis
    # Calculates outliers based on total sample intensity (Total Ion Current proxy)
    print("\n--- 3. Z-Score Analysis ---")
    z_scores, std_dev = detector.calculate_z_scores(df_neg_raw)

    # Print summary
    print(f"Z-Score Stats -> Mean: {z_scores.mean():.4f}, Std Dev (Intensity): {std_dev:.4f}")

    # Identify and print potential outliers (e.g., > 3 sigma)
    outliers_z = z_scores[np.abs(z_scores) > 3]
    if not outliers_z.empty:
        print(f"Potential Intensity Outliers (>3 sigma): {len(outliers_z)}")
        print(outliers_z.index.tolist())
    else:
        print("No extreme intensity outliers (>3 sigma) detected")
    # 5. Benchmarking
    # Runs all algorithms on the full dataset and scores them using Silhouette Score
    print("\n--- 4. Benchmarking Algorithms ---")
    benchmark_results = detector.benchmark_algorithms(df_neg_raw, fname="Negative Dataset")
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