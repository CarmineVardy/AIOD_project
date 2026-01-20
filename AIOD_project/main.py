"""
Main Execution Pipeline for the AIOD Metabolomics Project.

This script orchestrates the entire workflow, from raw data loading to final biomarker validation.
The process is structured in sequential phases:
1. Data Loading & Cleaning
2. Quality Assessment (Raw Data)
3. Pre-processing Strategy Evaluation (Normalization, Transformation, Scaling)
4. Low-Level Data Fusion (ESI- and ESI+)
5. Anomaly Detection (Statistical & ML Consensus)
6. Dataset Refinement (Outlier Removal)
7. Data Splitting (Prevention of Data Leakage)
8. Model Training & Feature Selection (Lasso, PLS-DA)
9. Final Biomarker Validation

Note: this is not a static pipeline that we applied. It is only a report of flow necessary operation for each phase

Group 2: Carmine Vardaro, Marco savastano
Course: Artificial Intelligence for Omics Data Analysis
"""

import os
import pandas as pd
import numpy as np

# Import custom modules
from src.analysis import *
from src.anomalyDetection import *
from src.data_loader import *
from src.dataFusion import *
from src.datasetSplitting import *
from src.models import *
from src.pca import *
from src.plsDa import *
from src.preprocessing import *
from src.visualization import *
from src.datasetSplitting import *

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_PATH, "data")
OUT_PATH = os.path.join(PROJECT_PATH, "out")

# Input Paths
RAW_DIR = os.path.join(DATA_PATH, "raw")
NEG_RAW_PATH = os.path.join(RAW_DIR, '2024_Metabolomica_Neg.csv')
POS_RAW_PATH = os.path.join(RAW_DIR, '2024_Metabolomica_Pos.csv')

TRANSPOSED_DIR = os.path.join(DATA_PATH, "transposed")
NEG_TRANSPOSED_PATH = os.path.join(TRANSPOSED_DIR, '2024_Metabolomica_Neg_Transposed.csv')
POS_TRANSPOSED_PATH = os.path.join(TRANSPOSED_DIR, '2024_Metabolomica_Pos_Transposed.csv')

CLEANED_DIR = os.path.join(DATA_PATH, "cleaned")
NEG_CLEANED_PATH = os.path.join(CLEANED_DIR, '2024_Metabolomica_Neg_Cleaned.csv')
POS_CLEANED_PATH = os.path.join(CLEANED_DIR, '2024_Metabolomica_Pos_Cleaned.csv')


def main():
    os.makedirs(OUT_PATH, exist_ok=True)

    # -------------------------------------------------------------------------
    # DATA LOADING AND PREPARATION
    # -------------------------------------------------------------------------
    # If intermediate files don't exist, generate them from raw data
    if not os.path.exists(NEG_CLEANED_PATH):
        print("Generating transposed and cleaned datasets...")
        os.makedirs(TRANSPOSED_DIR, exist_ok=True)
        os.makedirs(CLEANED_DIR, exist_ok=True)

        # Reshape (Transpose)
        reshape_dataset(load_raw_dataset(NEG_RAW_PATH)).to_csv(NEG_TRANSPOSED_PATH, index=True)
        reshape_dataset(load_raw_dataset(POS_RAW_PATH)).to_csv(POS_TRANSPOSED_PATH, index=True)

        # Clean (Remove QCs and Tech Replicates)
        remove_qc_and_technical_replicates(load_transposed_dataset(NEG_TRANSPOSED_PATH)).to_csv(NEG_CLEANED_PATH,
                                                                                                index=True)
        remove_qc_and_technical_replicates(load_transposed_dataset(POS_TRANSPOSED_PATH)).to_csv(POS_CLEANED_PATH,
                                                                                                index=True)

    # Load Datasets
    df_neg_raw = load_transposed_dataset(NEG_TRANSPOSED_PATH)  # Contains QCs/Replicates
    df_pos_raw = load_transposed_dataset(POS_TRANSPOSED_PATH)
    df_neg = load_transposed_dataset(NEG_CLEANED_PATH)  # Biological samples only
    df_pos = load_transposed_dataset(POS_CLEANED_PATH)

    # -------------------------------------------------------------------------
    # QUALITY ASSESSMENT (RAW DATA)
    # -------------------------------------------------------------------------
    qa_dir = os.path.join(OUT_PATH, "quality_assessment", "pca")
    os.makedirs(qa_dir, exist_ok=True)

    # PCA on Raw Data (Negative Mode)
    pca_res_neg = perform_pca(df_neg_raw, scaling='autoscaling')
    plot_pca_scores(pca_res_neg, df_neg_raw, qa_dir, file_name="pca_neg_raw", class_col="Class", show_ellipse=False)

    # QC Visualization (Highlighting QCs)
    df_qc_viz = df_neg_raw.copy()
    df_qc_viz['Class'] = np.where(df_neg_raw['Class'] == 'QC', 'QC', 'NotShow')
    plot_pca_scores(pca_res_neg, df_qc_viz, qa_dir, file_name="pca_neg_qc_check", class_col="Class", show_ellipse=False)

    # Replicates Visualization
    df_rep_viz = df_neg_raw.copy()
    df_rep_viz['Class'] = 'NotShow'
    replicates = df_neg_raw.index[df_neg_raw.index.str.contains('_01', regex=False)]
    base_names = set([x.split('_01')[0] for x in replicates])
    for name in base_names:
        mask = df_rep_viz.index.str.contains(name, regex=False)
        df_rep_viz.loc[mask, 'Class'] = name
    plot_pca_scores(pca_res_neg, df_rep_viz, qa_dir, file_name="pca_neg_replicates_check", class_col="Class",
                    show_ellipse=False)

    # -------------------------------------------------------------------------
    # PRE-PROCESSING STRATEGY EVALUATION
    # -------------------------------------------------------------------------
    pp_dir = os.path.join(OUT_PATH, "pre_processing")

    # A. Normalization Comparison (Calculating CV%)
    norm_methods = {
        "tic": normalization_tic, "median": normalization_median,
        "mean": normalization_mean, "pqn": normalization_pqn, "quantile": normalization_quantile
    }

    for mode, df_data in [("neg", df_neg), ("pos", df_pos)]:
        mode_dir = os.path.join(pp_dir, mode, "normalization")
        os.makedirs(mode_dir, exist_ok=True)
        results_list = []

        for name, func in norm_methods.items():
            df_norm = func(df_data)
            # Calculate CV for CTRL and CHD
            cv_ctrl = compute_feature_statistics(df_norm[df_norm["Class"] == "CTRL"])["cv_percent"].median()
            cv_chd = compute_feature_statistics(df_norm[df_norm["Class"] == "CHD"])["cv_percent"].median()
            results_list.append({"Method": name, "Avg_CV": (cv_ctrl + cv_chd) / 2})

            # Plot Sample Distributions
            plot_sample_distributions(df_norm, mode_dir, file_name=name, class_col="Class", samples_per_page=50)

        pd.DataFrame(results_list).sort_values("Avg_CV").to_csv(os.path.join(mode_dir, "cv_comparison.csv"),
                                                                index=False)

    # B. Transformation Evaluation (Density Plots)
    trans_methods = {
        "no_transf": (lambda x: x, "Raw"),
        "log_10": (transformation_log10, "Log10")
    }

    for mode, df_data in [("neg", df_neg), ("pos", df_pos)]:
        trans_dir = os.path.join(pp_dir, mode, "transformation")
        os.makedirs(trans_dir, exist_ok=True)

        for name, (func, label) in trans_methods.items():
            plot_global_density(func(df_data), trans_dir, file_name=name, xlabel=label, add_gaussian=True)

    # C. Scaling Evaluation (Boxplots Overview)
    for mode, df_data in [("neg", df_neg), ("pos", df_pos)]:
        scale_dir = os.path.join(pp_dir, mode, "scaling")
        os.makedirs(scale_dir, exist_ok=True)

        plot_features_overview(df_data, scale_dir, file_name="no_autoscaling")
        plot_features_overview(scaling_autoscaling(df_data), scale_dir, file_name="with_autoscaling")

    # --- APPLY SELECTED PRE-PROCESSING (For Data Fusion) ---
    # Selected: PQN -> Log10 -> Autoscaling (Performed implicitly in Low Level Fusion if needed,
    # but here we do PQN+Log10 before Fusion, and Scaling inside Fusion/Detection).

    df_neg_proc = transformation_log10(normalization_pqn(df_neg))
    df_pos_proc = transformation_log10(normalization_pqn(df_pos))

    # We apply autoscaling now for the Anomaly Detection phase
    df_neg_scaled = scaling_autoscaling(df_neg_proc)
    df_pos_scaled = scaling_autoscaling(df_pos_proc)

    # -------------------------------------------------------------------------
    # LOW-LEVEL DATA FUSION
    # -------------------------------------------------------------------------
    fusion_dir = os.path.join(OUT_PATH, "data_fusion", "sum_pca")
    os.makedirs(fusion_dir, exist_ok=True)

    # Fusion involves Block Scaling (Frobenius) then Concatenation
    df_fused = low_level_fusion([df_neg_scaled, df_pos_scaled])

    # SUM-PCA on Fused Data
    sum_pca_res = perform_pca(df_fused, scaling=None)  # Data already scaled
    plot_pca_scores(sum_pca_res, df_fused, fusion_dir, file_name="sum_pca_score_plot", is_sum_pca=True)
    plot_sum_pca_loadings(sum_pca_res, fusion_dir, pc_x=1, pc_y=2, file_name="sum_pca_loadings")
    plot_sum_pca_scree_contribution(sum_pca_res, fusion_dir, file_name="sum_pca_scree_blocks")

    # -------------------------------------------------------------------------
    # ANOMALY DETECTION
    # -------------------------------------------------------------------------
    ad_dir = os.path.join(OUT_PATH, "anomaly_detection")

    # Analysis per class (CHD vs CTRL) on Fused Data
    for cls in ["CTRL", "CHD"]:
        cls_dir = os.path.join(ad_dir, "pca_outliers")
        os.makedirs(cls_dir, exist_ok=True)

        df_cls = df_fused[df_fused["Class"] == cls]
        pca_res_cls = perform_pca(df_cls, scaling=None)

        # 1. Statistical Detection (T2 / Q)
        outliers_stat = detect_pca_outliers(pca_res_cls)
        outliers_stat[outliers_stat["Outlier_Type"] != "Normal"].to_csv(
            os.path.join(cls_dir, f"{cls}_stat_outliers.csv"))
        plot_distance_plot(pca_res_cls, df_cls, os.path.join(ad_dir, "distance_plots"), file_name=cls)

        # 2. ML Detection (Visualization only here, actual voting logic assumed done)
        plot_anomaly_comparison(df_cls.drop(columns=['Class']), f"{cls}_Group", os.path.join(ad_dir, "ml_outliers"))

        # 3. Visual Inspection
        plot_sample_distributions(df_cls, os.path.join(ad_dir, "box_plots"), file_name=f"{cls}_samples_qc_boxplot")

    # -------------------------------------------------------------------------
    # DATASET REFINEMENT (OUTLIER REMOVAL)
    # -------------------------------------------------------------------------
    # Consensus Outliers to Remove (Identified via T2/Q + ML Consensus + Boxplots)
    def_outliers_list = [
        "CTRL02_00 (F249)", "CTRL09_00 (F258)", "CTRL41 (F290)",
        "CTRL53 (F304)", "CTRL60 (F311)", "CTRL93_00 (F344)",
        "P06_00 (F375)", "P42 (F411)", "P59 (F430)", "P93 (F464)",
    ]

    # Reload Original Data to restart processing WITHOUT outliers
    df_neg_clean = load_transposed_dataset(NEG_CLEANED_PATH)
    df_pos_clean = load_transposed_dataset(POS_CLEANED_PATH)

    # Drop Outliers
    df_neg_clean = df_neg_clean.drop(index=[x for x in def_outliers_list if x in df_neg_clean.index])
    df_pos_clean = df_pos_clean.drop(index=[x for x in def_outliers_list if x in df_pos_clean.index])

    # Re-apply PQN + Log10 (Scaling will be done inside Cross-Validation)
    df_neg_final = transformation_log10(normalization_pqn(df_neg_clean))
    df_pos_final = transformation_log10(normalization_pqn(df_pos_clean))

    # Prepare Raw Concatenation for Splitting Indices generation
    raw_data_fusion = pd.concat([df_neg_final, df_pos_final.drop(columns=['Class'])], axis=1)

    # -------------------------------------------------------------------------
    # DATA SPLITTING (PREVENTION OF DATA LEAKAGE)
    # -------------------------------------------------------------------------
    training_dir = os.path.join(OUT_PATH, "split_and_training")

    # Define Strategies
    split_strategies = {
        "Random_KFold_CV": get_random_kfold_cv,
        "Stratified_KFold_CV": get_stratified_kfold_cv,
        "Duplex": get_duplex_split
    }

    final_datasets = {}

    # Generate Indices on Raw Data
    for name, func in split_strategies.items():
        folds = []
        if "Duplex" in name:
            # Single Split for Duplex
            X_tr, X_te, _, _ = func(raw_data_fusion, split_ratio=0.75, use_pca=True, perform_scaling=False)
            folds.append((X_tr.index, X_te.index))
        else:
            # 5-Fold CV for Random/Stratified
            cv_folds = func(raw_data_fusion, n_splits=5, perform_scaling=False)
            folds = [(f[0].index, f[1].index) for f in cv_folds]

        final_datasets[name] = []

        # Apply Anti-Leakage Transformation per Fold
        for train_idx, test_idx in folds:
            # 1. Slice Data
            X_tr_n, X_te_n = df_neg_final.loc[train_idx].drop(columns=['Class']), df_neg_final.loc[test_idx].drop(
                columns=['Class'])
            X_tr_p, X_te_p = df_pos_final.loc[train_idx].drop(columns=['Class']), df_pos_final.loc[test_idx].drop(
                columns=['Class'])
            y_tr, y_te = df_neg_final.loc[train_idx]['Class'], df_neg_final.loc[test_idx]['Class']

            # 2. Learn Scaling on Train -> Apply to Test
            X_tr_n, X_te_n = apply_scaling(X_tr_n, X_te_n)
            X_tr_p, X_te_p = apply_scaling(X_tr_p, X_te_p)

            # 3. Learn Block Weighting (Frobenius) on Train -> Apply to Test
            X_tr_n, X_te_n = apply_block_weighting_split(X_tr_n, X_te_n, 0)
            X_tr_p, X_te_p = apply_block_weighting_split(X_tr_p, X_te_p, 1)

            # 4. Fuse
            final_datasets[name].append({
                'X_train': pd.concat([X_tr_n, X_tr_p], axis=1),
                'X_test': pd.concat([X_te_n, X_te_p], axis=1),
                'y_train': y_tr, 'y_test': y_te
            })

    # Evaluate Splits Quality (Metrics)
    metrics_list = []
    for name, folds in final_datasets.items():
        for i, fold in enumerate(folds):
            metrics_list.append(evaluate_split_quality(
                fold['X_train'], fold['X_test'], fold['y_train'], fold['y_test'],
                split_name=f"{name}_{i + 1}"
            ))
    pd.DataFrame(metrics_list).to_csv(os.path.join(training_dir, "split", "splitting_metrics_summary.csv"), index=False)

    # -------------------------------------------------------------------------
    # MODEL TRAINING & FEATURE SELECTION
    # -------------------------------------------------------------------------
    lasso_dir = os.path.join(training_dir, "training", "last", "LASSO")
    pls_dir = os.path.join(training_dir, "training", "last", "PLS-DA")
    os.makedirs(lasso_dir, exist_ok=True)
    os.makedirs(pls_dir, exist_ok=True)

    stratified_data = final_datasets["Stratified_KFold_CV"]
    lasso_feats, pls_feats = [], []

    for i, fold in enumerate(stratified_data):
        X_tr, y_tr, X_te, y_te = fold['X_train'], fold['y_train'], fold['X_test'], fold['y_test']
        fold_id = f"Fold_{i + 1}"

        # A. LASSO (Feature Selection)
        _, _, info_lasso = run_logistic_regression(X_tr, y_tr, X_te, penalty='l1', C=50, solver='liblinear')
        for feat, coef in info_lasso['feature_importance_ranking'].items():
            lasso_feats.append(
                {'Fold': fold_id, 'Metabolite': feat, 'Coefficient': coef, 'Selected': 1 if coef != 0 else 0})

        # B. PLS-DA (Optimization & Training)
        # Optimization Loop
        errors = []
        for n in range(1, 11):
            y_p, y_prob, _, _ = run_pls_da(X_tr, y_tr, X_te, n_components=n)
            metrics = compute_classification_metrics(y_te, y_p, y_prob)
            errors.append({'n_components': n, 'Error': metrics['Classification_Error']})

        best_n = int(pd.DataFrame(errors).sort_values("Error").iloc[0]['n_components'])

        # Final PLS-DA
        y_p, y_prob, info_pls, res_pls = run_pls_da(X_tr, y_tr, X_te, n_components=best_n)
        for feat, vip in info_pls['feature_importance_ranking'].items():
            pls_feats.append({'Fold': fold_id, 'Metabolite': feat, 'VIP': vip})

        # Permutation Test (Validation)
        run_permutation_test(X_tr, y_tr, n_components=best_n, n_permutations=100)

        # Plotting (First Fold only for report)
        if i == 0:
            plot_pls_scores(res_pls, pls_dir, file_name=f"pls_scores_{fold_id}")
            plot_pls_loadings(res_pls, pls_dir, file_name=f"pls_loadings_{fold_id}")
            plot_pls_predicted_vs_observed(y_te, y_prob, pls_dir, file_name=f"pls_pred_vs_obs_{fold_id}")

    # Save Feature Selection Results
    pd.DataFrame(lasso_feats).to_csv(os.path.join(lasso_dir, "lasso_feature_selection.csv"), index=False)
    pd.DataFrame(pls_feats).to_csv(os.path.join(pls_dir, "plsda_vip_scores.csv"), index=False)

    # -------------------------------------------------------------------------
    # FINAL BIOMARKER VALIDATION
    # -------------------------------------------------------------------------
    final_dir = os.path.join(OUT_PATH, "final")
    os.makedirs(final_dir, exist_ok=True)

    # Consensus Features (Hardcoded based on Lasso intersection PLS VIP > 1.5)
    final_features = [
        '4-Hydroxybutyric acid (GHB)', '5-Hydroxy-DL-tryptophan', 'DL-Lactic Acid',
        'Palmitic Acid', 'Linoleamide', 'Acetyl-L-carnitine', 'L-Cystine', 'Myristamide'
    ]

    # Prepare Final Dataset (Reduced to 8 features)
    df_neg_final_red = df_neg_final[['Class'] + [f for f in final_features if f in df_neg_final.columns]]
    df_pos_final_red = df_pos_final[['Class'] + [f for f in final_features if f in df_pos_final.columns]]

    # Merge for Descriptive Stats (Univariate / Boxplots) - No Block Scaling needed here for visualization
    df_final_merged = pd.concat([df_neg_final_red, df_pos_final_red.drop(columns=['Class'])], axis=1)

    # 1. Univariate Analysis
    perform_univariate_analysis(df_final_merged, class_col='Class').to_csv(
        os.path.join(final_dir, "univariate_results.csv"))

    # 2. Boxplots
    plot_boxplots(df_final_merged, final_features, final_dir, file_prefix="boxplots_final", features_per_page=8)

    # 3. Final PCA (Scaled)
    df_final_scaled = scaling_autoscaling(df_final_merged)
    pca_final = perform_pca(df_final_scaled, scaling=None)
    plot_pca_scores(pca_final, df_final_scaled, final_dir, file_name="pca_scores_final")
    plot_pca_loadings(pca_final, final_dir, top_n=8)
    plot_pca_scree(pca_final, final_dir)

    print("\nPipeline Completed Successfully.")


if __name__ == "__main__":
    main()