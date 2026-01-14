import os
import pandas as pd

import src.analysis
from src.analysis import *
from src.anomalyDetection import *
from src.data_loader import *
from src.dataFusion import *
from src.evaluation import *
from src.models import *
from src.preprocessing import *
from src.univariate_analysis import *
from src.visualization import *

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