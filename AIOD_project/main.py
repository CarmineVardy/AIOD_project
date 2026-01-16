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

def main():

    df_neg_raw = load_transposed_dataset(NEG_TRANSPOSED_PATH)
    df_pos_raw = load_transposed_dataset(POS_TRANSPOSED_PATH)
    df_neg = load_transposed_dataset(NEG_CLEANED_PATH)
    df_pos = load_transposed_dataset(POS_CLEANED_PATH)

    os.makedirs(OUT_PATH, exist_ok=True)


    # -------------------------------
    #     START QUALITY ASSESSMENT
    # -------------------------------
    qual_ass_base_dir = os.path.join(OUT_PATH, "quality_assessment")
    qual_ass_dirs = [
        qual_ass_base_dir,
        os.path.join(qual_ass_base_dir, "pca"),
        os.path.join(qual_ass_base_dir, "pca", "neg"),
        os.path.join(qual_ass_base_dir, "pca", "pos"),
    ]
    for d in qual_ass_dirs:
        os.makedirs(d, exist_ok=True)

    # --- Negative Mode Analysis ---
    pca_results_neg = perform_pca(df_neg_raw, scaling = 'autoscaling')
    plot_pca_loadings(pca_results_neg, qual_ass_dirs[2], pc_x=1, pc_y=2, top_n=0)
    plot_pca_scree(pca_results_neg, qual_ass_dirs[2], threshold=0.9)
    plot_loading_profile(pca_results_neg, qual_ass_dirs[2], pc_index=1)
    plot_loading_profile(pca_results_neg, qual_ass_dirs[2], pc_index=2)
    plot_pca_scores(pca_results_neg, df_neg_raw, qual_ass_dirs[2], pc_x=1, pc_y=2, file_name="pca_score_plot",class_col="Class", show_ellipse=False)

    # Plot for QC visualization
    plot_pca_scores(pca_results_neg, df_neg_raw.copy().assign(Class=np.where(df_neg_raw['Class'] == 'QC', 'QC', 'NotShow')), qual_ass_dirs[2], pc_x=1, pc_y=2, file_name="pca_score_plot_for_qc", class_col= "Class", show_ellipse= False)

    # Plot for REPLICATES visualization
    df_neg_raw_replicates_viz = df_neg_raw.copy()
    df_neg_raw_replicates_viz['Class'] = 'NotShow'
    mask = df_neg_raw_replicates_viz.index.str.contains('_01', regex=False)
    replicates_01 = df_neg_raw_replicates_viz.index[mask]
    base_names_clean = [x.split('_01')[0] for x in replicates_01]
    for name in set(base_names_clean):
        mask = df_neg_raw_replicates_viz.index.str.contains(name, regex=False)
        df_neg_raw_replicates_viz.loc[mask, 'Class'] = name
    plot_pca_scores(pca_results_neg, df_neg_raw_replicates_viz, qual_ass_dirs[2], pc_x=1, pc_y=2, file_name="pca_score_plot_for_replicates", class_col="Class", show_ellipse=False)

    # --- Positive Mode Analysis ---
    pca_results_pos = perform_pca(df_pos_raw, scaling='autoscaling')
    plot_pca_loadings(pca_results_pos, qual_ass_dirs[3], pc_x=1, pc_y=2, top_n=0)
    plot_pca_scree(pca_results_pos, qual_ass_dirs[3], threshold=0.9)
    plot_loading_profile(pca_results_pos, qual_ass_dirs[3], pc_index=1)
    plot_loading_profile(pca_results_pos, qual_ass_dirs[3], pc_index=2)
    plot_pca_scores(pca_results_pos, df_pos_raw, qual_ass_dirs[3], pc_x=1, pc_y=2, file_name="pca_score_plot", class_col= "Class", show_ellipse= False)

    # Plot for QC visualization
    plot_pca_scores(pca_results_pos, df_pos_raw.copy().assign(Class=np.where(df_pos_raw['Class'] == 'QC', 'QC', 'NotShow')), qual_ass_dirs[3], pc_x=1, pc_y=2, file_name="pca_score_plot_for_qc", class_col= "Class", show_ellipse= False)

    # Plot for REPLICATES visualization
    df_pos_raw_replicates_viz = df_pos_raw.copy()
    df_pos_raw_replicates_viz.drop(columns=['Class'], inplace=True)
    df_pos_raw_replicates_viz['Sample'] = df_neg_raw_replicates_viz['Class']
    plot_pca_scores(pca_results_pos, df_pos_raw_replicates_viz, qual_ass_dirs[3], pc_x=1, pc_y=2, file_name="pca_score_plot_for_replicates", class_col="Sample", show_ellipse=False)

    # -------------------------------
    #     END QUALITY ASSESSMENT
    # -------------------------------


    # -------------------------------
    #     START PRE-PROCESSING
    # -------------------------------
    pre_process_base_dir = os.path.join(OUT_PATH, "pre_processing")
    pre_process_dirs = [
        pre_process_base_dir,
        os.path.join(pre_process_base_dir, "neg"),
        os.path.join(pre_process_base_dir, "neg", "normalization"),
        os.path.join(pre_process_base_dir, "neg", "transformation"),
        os.path.join(pre_process_base_dir, "neg", "scaling"),
        os.path.join(pre_process_base_dir, "pos"),
        os.path.join(pre_process_base_dir, "pos", "normalization"),
        os.path.join(pre_process_base_dir, "pos", "transformation"),
        os.path.join(pre_process_base_dir, "pos", "scaling"),
        os.path.join(pre_process_base_dir, "neg", "pca"),
        os.path.join(pre_process_base_dir, "pos", "pca"),
    ]

    for d in pre_process_dirs:
        os.makedirs(d, exist_ok=True)


    #NORMALIZATION
    normalization_methods = {
        "no_norm": lambda x: x.copy(),
        "tic": normalization_tic,
        "median": normalization_median,
        "mean": normalization_mean,
        "max": normalization_max,
        "range": normalization_range,
        "pqn": normalization_pqn,
        "quantile": normalization_quantile
    }
    datasets_config = {
        "neg": {"data": df_neg, "out_dir": pre_process_dirs[2]},
        "pos": {"data": df_pos, "out_dir": pre_process_dirs[6]}
    }
    comparison_tables = {}
    for mode, config in datasets_config.items():
        df_raw = config["data"]
        output_dir = config["out_dir"]
        results_list = []
        for method_name, norm_func in normalization_methods.items():
            df_normalized = norm_func(df_raw).copy()
            plot_title = "" if method_name != "no_norm" else None
            plot_sample_distributions(
                df_normalized,
                output_dir=output_dir,
                file_name=method_name,
                class_col="Class",
                samples_per_page=197,
                plot_title=plot_title
            )
            df_ctrl = df_normalized[df_normalized["Class"] == "CTRL"].copy()
            stats_ctrl = compute_feature_statistics(df_ctrl)
            cv_ctrl = stats_ctrl["cv_percent"].median()

            df_chd = df_normalized[df_normalized["Class"] == "CHD"].copy()
            stats_chd = compute_feature_statistics(df_chd)
            cv_chd = stats_chd["cv_percent"].median()

            results_list.append({
                "Normalization": method_name,
                "Median CV CTRL %": round(cv_ctrl, 2),
                "Median CV CHD %": round(cv_chd, 2),
                "Average CV %": round((cv_ctrl + cv_chd) / 2, 2)
            })

        df_comparison = pd.DataFrame(results_list).sort_values(by="Average CV %")
        comparison_tables[mode] = df_comparison

    df_comparison_neg = comparison_tables["neg"]
    df_comparison_pos = comparison_tables["pos"]
    out_path_neg = os.path.join(pre_process_base_dir, "neg", "normalization", "normalization_comparison.csv")
    df_comparison_neg.to_csv(out_path_neg, index=False)
    out_path_pos = os.path.join(pre_process_base_dir, "pos", "normalization", "normalization_comparison.csv")
    df_comparison_pos.to_csv(out_path_pos, index=False)

    #TRANSFORMATIONS
    transformations = {
        "no_transf": (lambda x: x, "Raw Intensity"),
        "log_10": (transformation_log10, "Log10 Intensity"),
        "log_2": (transformation_log2, "Log2 Intensity"),
        "log_e": (transformation_log_e, "Log_e Intensity"),
        "log_sqrt": (transformation_sqrt, "Log_sqrt Intensity"),
        "log_cuberoot": (transformation_cuberoot, "Log_cuberoot Intensity")
    }

    datasets_config = [
        (df_neg, pre_process_dirs[3]),
        (df_pos, pre_process_dirs[7])
    ]

    for df_source, out_dir in datasets_config:
        for file_name, (func, label) in transformations.items():
            plot_global_density(
                df=func(df_source),
                output_dir=out_dir,
                file_name=file_name,
                class_col='Class',
                xlabel=label,
                add_gaussian=True,
            )

    #SCALING
    plot_features_overview(df_neg, output_dir = pre_process_dirs[4], file_name="no_autoscaling", class_col='Class', features_per_page=52,plot_title=None)
    plot_features_overview(scaling_autoscaling(df_neg), output_dir = pre_process_dirs[4], file_name="with_autoscaling", class_col='Class', features_per_page=52,plot_title='')

    plot_features_overview(df_pos, output_dir = pre_process_dirs[8], file_name="no_autoscaling", class_col='Class', features_per_page=98,plot_title=None)
    plot_features_overview(scaling_autoscaling(df_pos), output_dir = pre_process_dirs[8], file_name="with_autoscaling", class_col='Class', features_per_page=98,plot_title='')



    # -------------------------------
    #     END PRE-PROCESSING
    # -------------------------------

    '''

    plots_dir = os.path.join(STEP2_DIR, "plots")

    """#plot_class_distribution_pie(df_neg_raw, plots_dir, "Datset Negative raw")
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
    internal_variability(df_pos_raw, "Internal variability positive Raw", plots_dir)"""


# ==============================================================================
#  ANOMALIES DETECTION
# ==============================================================================

    detector = AnomalyDetector() 

    """# 2. Visualizing Mahalanobis Contours
    # This method creates a specific plot for Robust Covariance
    print("\n--- 1. Mahalanobis Distance Analysis ---")
    detector.plot_mahalanobis_contours(df_neg_raw, "Negative Dataset - Mahalanobis Distance Contours", plots_dir)
    # 3. Visualizing Algorithm Grid
    # This method compares multiple algorithms on a 2D grid.
    # We must project the data to 2D (PCA) before passing it to the grid plotter.
    print("\n--- 2. Algorithm Comparison Grid ---")
    X_pca_2d, _ = detector._generate_pca(df_neg_raw, n_components=2)
    detector.plot_anomaly_grid(
        X_pca_2d,         
        "Negative Dataset Anomaly Grid", 
        plots_dir,
        detector.algorithms
    )"""
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

    df_neg_clean = detector.remove_outliers(df_neg_raw, outliers_z)

    biplot(df_neg_raw, "Negative Raw", plots_dir)
    biplot(df_neg_clean, "Negative anomaly cleaned", plots_dir)
    
    # 5. Benchmarking
    # Runs all algorithms on the full dataset and scores them using Silhouette Score
    print("\n--- 4. Benchmarking Algorithms ---")
    benchmark_results = detector.benchmark_algorithms(df_neg_raw, fname="Negative Dataset")
    # The benchmark method prints the table automatically, but it is also returned here

    consensus_outliers = detector.identify_consensus_outliers(df_neg_raw, fname="Negative Dataset")

    detector.identify_consensus_outliers(df_neg_raw, fname="Negative Dataset")

    df_neg_clean_2 = detector.remove_outliers(df_neg_raw, consensus_outliers)

    biplot(df_neg_clean_2, "Negative anomaly cleaned 2", plots_dir)

'''

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