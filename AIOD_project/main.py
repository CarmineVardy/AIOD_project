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
    ]
    for d in pre_process_dirs:
        os.makedirs(d, exist_ok=True)

    # NORMALIZATION NEG

    #WITHOUT NORMALIZATION
    plot_sample_distributions(df_neg, output_dir=pre_process_dirs[2], file_name="no_norm", class_col="Class", samples_per_page=197)

    plot_sample_distributions(normalization_tic(df_neg), output_dir=pre_process_dirs[2], file_name="tic", class_col="Class", samples_per_page=197, plot_title='')
    plot_sample_distributions(normalization_median(df_neg), output_dir=pre_process_dirs[2], file_name="median", class_col="Class", samples_per_page=197, plot_title='')
    plot_sample_distributions(normalization_mean(df_neg), output_dir=pre_process_dirs[2], file_name="mean", class_col="Class", samples_per_page=197, plot_title='')
    plot_sample_distributions(normalization_max(df_neg), output_dir=pre_process_dirs[2], file_name="max", class_col="Class", samples_per_page=197, plot_title='')
    plot_sample_distributions(normalization_range(df_neg), output_dir=pre_process_dirs[2], file_name="range", class_col="Class", samples_per_page=197, plot_title='')
    plot_sample_distributions(normalization_pqn(df_neg), output_dir=pre_process_dirs[2], file_name="pqn", class_col="Class", samples_per_page=197, plot_title='')
    plot_sample_distributions(normalization_quantile(df_neg), output_dir=pre_process_dirs[2], file_name="quantile", class_col="Class", samples_per_page=197, plot_title='')

    # NORMALIZATION POS

    # WITHOUT NORMALIZATION
    plot_sample_distributions(df_pos, output_dir=pre_process_dirs[6], file_name= "no_norm", class_col= "Class", samples_per_page=197)

    plot_sample_distributions(normalization_tic(df_pos), output_dir=pre_process_dirs[6], file_name= "tic", class_col= "Class", samples_per_page=197, plot_title='')
    plot_sample_distributions(normalization_median(df_pos), output_dir=pre_process_dirs[6], file_name= "median", class_col= "Class", samples_per_page=197, plot_title='')
    plot_sample_distributions(normalization_mean(df_pos), output_dir=pre_process_dirs[6], file_name= "mean", class_col= "Class", samples_per_page=197, plot_title='')
    plot_sample_distributions(normalization_max(df_pos), output_dir=pre_process_dirs[6], file_name= "max", class_col= "Class", samples_per_page=197, plot_title='')
    plot_sample_distributions(normalization_range(df_pos), output_dir=pre_process_dirs[6], file_name= "range", class_col= "Class", samples_per_page=197, plot_title='')
    plot_sample_distributions(normalization_pqn(df_pos), output_dir=pre_process_dirs[6], file_name= "pqn", class_col= "Class", samples_per_page=197, plot_title='')
    plot_sample_distributions(normalization_quantile(df_pos), output_dir=pre_process_dirs[6], file_name= "quantile", class_col= "Class", samples_per_page=197, plot_title='')

    '''
    #TRANSFORMATION
    transformation_log10()
    transformation_log2()
    transformation_log_e()
    transformation_sqrt()
    transformation_cuberoot()

    #SCALING
    scaling_autoscaling()

    '''

    # -------------------------------
    #     END PRE-PROCESSING
    # -------------------------------



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