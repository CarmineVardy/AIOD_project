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

    '''
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
        os.path.join(pre_process_base_dir, "neg", "pca", "without_preprocessing"),
        os.path.join(pre_process_base_dir, "neg", "pca", "with_preprocessing"),
        os.path.join(pre_process_base_dir, "pos", "pca"),
        os.path.join(pre_process_base_dir, "pos", "pca", "without_preprocessing"),
        os.path.join(pre_process_base_dir, "pos", "pca", "with_preprocessing"),
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


    pca_results_neg = perform_pca(df_neg, scaling = 'autoscaling')
    plot_pca_scores(pca_results_neg, df_neg, pre_process_dirs[10], pc_x=1, pc_y=2, file_name="pca_score_plot",class_col="Class", show_ellipse=False)
    plot_pca_loadings(pca_results_neg, pre_process_dirs[10], pc_x=1, pc_y=2, top_n=10)
    plot_pca_scree(pca_results_neg, pre_process_dirs[10], threshold=0.9)

    pca_results_pos = perform_pca(df_pos, scaling = 'autoscaling')
    plot_pca_scores(pca_results_pos, df_pos, pre_process_dirs[13], pc_x=1, pc_y=2, file_name="pca_score_plot",class_col="Class", show_ellipse=False)
    plot_pca_loadings(pca_results_pos, pre_process_dirs[13], pc_x=1, pc_y=2, top_n=10)
    plot_pca_scree(pca_results_pos, pre_process_dirs[13], threshold=0.9)

    '''


    # application of preprocessing
    df_neg = normalization_pqn(df_neg)
    df_neg = transformation_log10(df_neg)
    df_neg = scaling_autoscaling(df_neg)

    df_pos = normalization_pqn(df_pos)
    df_pos = transformation_log10(df_pos)
    df_pos = scaling_autoscaling(df_pos)

    '''
    pca_results_neg = perform_pca(df_neg, scaling = None)
    plot_pca_scores(pca_results_neg, df_neg, pre_process_dirs[11], pc_x=1, pc_y=2, file_name="pca_score_plot",class_col="Class", show_ellipse=False)
    plot_pca_loadings(pca_results_neg, pre_process_dirs[11], pc_x=1, pc_y=2, top_n=10)
    plot_pca_scree(pca_results_neg, pre_process_dirs[11], threshold=0.9)

    pca_results_pos = perform_pca(df_pos, scaling = None)
    plot_pca_scores(pca_results_pos, df_pos, pre_process_dirs[14], pc_x=1, pc_y=2, file_name="pca_score_plot",class_col="Class", show_ellipse=False)
    plot_pca_loadings(pca_results_pos, pre_process_dirs[14], pc_x=1, pc_y=2, top_n=10)
    plot_pca_scree(pca_results_pos, pre_process_dirs[14], threshold=0.9)

    '''

    # -------------------------------
    #     END PRE-PROCESSING
    # -------------------------------


    #DATA FUSION
    data_fusion_base_dir = os.path.join(OUT_PATH, "data_fusion")
    data_fusion_dirs = [
        data_fusion_base_dir,
        os.path.join(data_fusion_base_dir, "sum_pca"),
    ]

    for d in data_fusion_dirs:
        os.makedirs(d, exist_ok=True)

    df_list = []
    df_list.append(df_neg)
    df_list.append(df_pos)
    df_low_level_merged = low_level_fusion(df_list)

    sum_pca_results = perform_pca(df_low_level_merged, scaling=None)

    '''
    plot_pca_scores(sum_pca_results, df_low_level_merged, data_fusion_dirs[1], pc_x= 1, pc_y= 2, file_name="sum_pca_score_plot", class_col="Class", show_ellipse= True, is_sum_pca=True)
    plot_sum_pca_loadings(sum_pca_results,output_dir=data_fusion_dirs[1], pc_x=1, pc_y=2, file_name="sum_pca_loadings_blocks", top_n=10)
    plot_pca_scores(sum_pca_results, df_low_level_merged, data_fusion_dirs[1], pc_x= 1, pc_y= 3, file_name="sum_pca_score_plot", class_col="Class", show_ellipse= True, is_sum_pca=True)
    plot_sum_pca_loadings(sum_pca_results,output_dir=data_fusion_dirs[1], pc_x=1, pc_y=3, file_name="sum_pca_loadings_blocks", top_n=10)
    plot_pca_scores(sum_pca_results, df_low_level_merged, data_fusion_dirs[1], pc_x= 2, pc_y= 3, file_name="sum_pca_score_plot", class_col="Class", show_ellipse= True, is_sum_pca=True)
    plot_sum_pca_loadings(sum_pca_results,output_dir=data_fusion_dirs[1], pc_x=2, pc_y=3, file_name="sum_pca_loadings_blocks", top_n=10)
    plot_sum_pca_scree_contribution(sum_pca_results, output_dir=data_fusion_dirs[1], file_name="sum_pca_scree_blocks", threshold=0.9 )
    '''

    #ANOMALY DETECTION
    anomaly_detection_base_dir = os.path.join(OUT_PATH, "anomaly_detection")
    anomaly_detection_dirs = [
        anomaly_detection_base_dir,
        os.path.join(anomaly_detection_base_dir, "pca_outliers"),
        os.path.join(anomaly_detection_base_dir, "ml_outliers"),
        os.path.join(anomaly_detection_base_dir, "distance_plots"),
        os.path.join(anomaly_detection_base_dir, "box_plots"),
        os.path.join(anomaly_detection_base_dir, "pca_without_out")
    ]

    for d in anomaly_detection_dirs:
        os.makedirs(d, exist_ok=True)

    df_ctrl = df_low_level_merged[df_low_level_merged["Class"] == "CTRL"]
    df_chd = df_low_level_merged[df_low_level_merged["Class"] == "CHD"]

    pca_results_ctrl = perform_pca(df_ctrl, scaling=None)
    pca_results_chd = perform_pca(df_chd, scaling=None)

    '''
    df_pca_ctrl_outliers = detect_pca_outliers(pca_results_ctrl, conf_level=0.95)
    df_pca_chd_outliers = detect_pca_outliers(pca_results_chd, conf_level=0.95)

    df_pca_ctrl_outliers[df_pca_ctrl_outliers["Outlier_Type"] != "Normal"].to_csv(os.path.join(anomaly_detection_dirs[1], "ctrl_outliers.csv"), index=True)
    df_pca_chd_outliers[df_pca_chd_outliers["Outlier_Type"] != "Normal"].to_csv(os.path.join(anomaly_detection_dirs[1], "chd_outliers.csv"), index=True)

    n_comps_ctrl = np.argmax(pca_results_ctrl['cumulative_variance'] >= 0.95) + 1
    res_if_ctrl = run_isolation_forest(df_ctrl.drop(columns=['Class']), contamination=0.05)
    res_ocsvm_ctrl = run_one_class_svm(pca_results_ctrl['scores'].iloc[:, :n_comps_ctrl].values, nu=0.05)
    res_lof_ctrl = run_local_outlier_factor(pca_results_ctrl['scores'].iloc[:, :n_comps_ctrl].values, contamination=0.05)

    df_res_ctrl = pd.DataFrame(index=df_ctrl.index)

    df_res_ctrl['IsoForest_Outlier'] = (res_if_ctrl['y_pred'] == -1)
    df_res_ctrl['OCSVM_Outlier'] = (res_ocsvm_ctrl['y_pred'] == -1)
    df_res_ctrl['LOF_Outlier'] = (res_lof_ctrl['y_pred'] == -1)

    df_res_ctrl['IsoForest_Score'] = res_if_ctrl['scores']
    df_res_ctrl['OCSVM_Score'] = res_ocsvm_ctrl['scores']
    df_res_ctrl['LOF_Score'] = res_lof_ctrl['scores']

    cols_check = ['IsoForest_Outlier', 'OCSVM_Outlier', 'LOF_Outlier']
    df_res_ctrl_filtered = df_res_ctrl[df_res_ctrl[cols_check].any(axis=1)].copy()

    df_res_ctrl_filtered['Votes'] = df_res_ctrl_filtered[cols_check].sum(axis=1)
    df_res_ctrl_filtered = df_res_ctrl_filtered.sort_values(by='Votes', ascending=False).drop(columns=['Votes'])

    df_res_ctrl_filtered.to_csv(os.path.join(anomaly_detection_dirs[2], "ML_Outliers_CTRL.csv"))

    n_comps_chd = np.argmax(pca_results_chd['cumulative_variance'] >= 0.95) + 1

    res_if_chd = run_isolation_forest(df_chd.drop(columns=['Class']))
    res_ocsvm_chd = run_one_class_svm(pca_results_chd['scores'].iloc[:, :n_comps_chd].values)
    res_lof_chd = run_local_outlier_factor(pca_results_chd['scores'].iloc[:, :n_comps_chd].values)

    df_res_chd = pd.DataFrame(index=df_chd.index)

    df_res_chd['IsoForest_Outlier'] = (res_if_chd['y_pred'] == -1)
    df_res_chd['OCSVM_Outlier'] = (res_ocsvm_chd['y_pred'] == -1)
    df_res_chd['LOF_Outlier'] = (res_lof_chd['y_pred'] == -1)

    df_res_chd['IsoForest_Score'] = res_if_chd['scores']
    df_res_chd['OCSVM_Score'] = res_ocsvm_chd['scores']
    df_res_chd['LOF_Score'] = res_lof_chd['scores']

    cols_check = ['IsoForest_Outlier', 'OCSVM_Outlier', 'LOF_Outlier']
    df_res_chd_filtered = df_res_chd[df_res_chd[cols_check].any(axis=1)].copy()

    df_res_chd_filtered['Votes'] = df_res_chd_filtered[cols_check].sum(axis=1)
    df_res_chd_filtered = df_res_chd_filtered.sort_values(by='Votes', ascending=False).drop(columns=['Votes'])

    df_res_chd_filtered.to_csv(os.path.join(anomaly_detection_dirs[2], "ML_Outliers_CHD.csv"))

    plot_anomaly_comparison(df_ctrl.drop(columns=['Class']),"CTRL_Group", anomaly_detection_dirs[2])

    plot_anomaly_comparison(df_chd.drop(columns=['Class']),"CHD_Group",anomaly_detection_dirs[2])
    

    outliers_ctrl_list = [
        "CTRL02_00 (F249)",
        "CTRL09_00 (F258)",
        "CTRL13 (F262)",
        "CTRL41 (F290)",
        "CTRL53 (F304)",
        "CTRL60 (F311)",
        "CTRL93_00 (F344)",
        "CTRL01_00 (F247)",
        "CTRL49_00 (F299)",
        "CTRL07_00 (F256)",
        "CTRL82 (F333)",
        "CTRL97 (F350)"
    ]

    outliers_chd_list = [
        "P06_00 (F375)",
        "P42 (F411)",
        "P59 (F430)",
        "P01_00 (F368)",
        "P03_00 (F371)",
        "P65 (F436)",
        "P93 (F464)",
        "P72 (F443)",
        "P02_00 (F370)",
        "P27 (F397)",
        "P64 (F435)",
        "P50 (F421)",
        "P81 (F452)"
    ]

    all_outliers_list = outliers_ctrl_list + outliers_chd_list

    plot_distance_plot(pca_results_ctrl, df_ctrl, anomaly_detection_dirs[3], highlight_samples=outliers_ctrl_list, file_name="CTRL")
    plot_distance_plot(pca_results_chd, df_chd, anomaly_detection_dirs[3], highlight_samples=outliers_chd_list, file_name="CHD")

    plot_sample_distributions(df_ctrl, anomaly_detection_dirs[4], file_name="ctrl_samples_qc_boxplot", class_col='Class',samples_per_page=50, plot_title="Sample Intensity Distributions - CTRL", show_sample_names=True, showfliers=True)
    plot_sample_distributions(df_chd, anomaly_detection_dirs[4], file_name="chd_samples_qc_boxplot", class_col='Class',samples_per_page=49, plot_title="Sample Intensity Distributions - CHD", show_sample_names=True, showfliers=True)

    '''

    def_outliers_list = [
        "CTRL02_00 (F249)",
        "CTRL09_00 (F258)",
        "CTRL41 (F290)",
        "CTRL53 (F304)",
        "CTRL60 (F311)",
        "CTRL93_00 (F344)",
        "P06_00 (F375)",
        "P42 (F411)",
        "P59 (F430)",
        "P93 (F464)",
    ]

    df_neg = load_transposed_dataset(NEG_CLEANED_PATH)
    df_pos = load_transposed_dataset(POS_CLEANED_PATH)

    df_neg = df_neg.drop(index=[x for x in def_outliers_list if x in df_neg.index])
    df_pos = df_pos.drop(index=[x for x in def_outliers_list if x in df_pos.index])

    df_neg = normalization_pqn(df_neg)
    df_neg = transformation_log10(df_neg)
    df_neg = scaling_autoscaling(df_neg)
    df_pos = normalization_pqn(df_pos)
    df_pos = transformation_log10(df_pos)
    df_pos = scaling_autoscaling(df_pos)
    df_low_level_merged = low_level_fusion([df_neg, df_pos])

    '''
    pca_results = perform_pca(df_low_level_merged, scaling=None)
    plot_pca_scores(pca_results, df_low_level_merged, anomaly_detection_dirs[5], pc_x=1, pc_y=2, file_name="pca_score_plot",class_col="Class", show_ellipse=True)
    plot_pca_loadings(pca_results, anomaly_detection_dirs[5], pc_x=1, pc_y=2, top_n=20)
    plot_pca_scree(pca_results, anomaly_detection_dirs[5], threshold=0.9)
    '''


    #SPLIT AND TRAINING
    training_base_dir = os.path.join(OUT_PATH, "split_and_training")
    training_dirs = [
        training_base_dir,
        os.path.join(training_base_dir, "split"),
        os.path.join(training_base_dir, "training"),
        #os.path.join(training_base_dir, "training", "pls-da"),
    ]

    for d in training_dirs:
        os.makedirs(d, exist_ok=True)

    #SPLIT
    split_strategies = {
        "Random_KFold_CV": get_random_kfold_cv,
        "Stratified_KFold_CV": get_stratified_kfold_cv,
        "Duplex": get_duplex_split
    }

    #applying different splitting
    split_results_log = {}
    for strategy_name, split_func in split_strategies.items():
        if "Duplex" in strategy_name:
            #qui possiamo iterare su n_pc
            X_train, X_test, y_train, y_test = split_func(df_low_level_merged, target_col='Class', split_ratio=0.75, use_pca=True, n_pc=None, perform_scaling=True)
            folds_list = [(X_train, X_test, y_train, y_test)]
        else:
            folds_list = split_func(df_low_level_merged, target_col='Class', n_splits=5, perform_scaling=True)
        split_results_log[strategy_name] = folds_list

    #metrics for each splitting
    all_metrics_summary = []
    for strategy_name, folds_list in split_results_log.items():
        for i, (X_tr, X_te, y_tr, y_te) in enumerate(folds_list):
            if "Duplex" in strategy_name:
                row_name = strategy_name
            else:
                row_name = f"{strategy_name}_Fold{i + 1}"
            metrics = evaluate_split_quality(X_tr, X_te, y_tr, y_te, split_name=row_name)
            all_metrics_summary.append(metrics)

    df_metrics_final = pd.DataFrame(all_metrics_summary)
    metrics_csv_path = os.path.join(training_dirs[1], "splitting_metrics_summary.csv")
    df_metrics_final.to_csv(metrics_csv_path, index=False)

    #TRAINING

    '''

    models_config = {
        "SVM": run_svm_model,
        "Logistic_Regression": run_logistic_regression,
        "Random_Forest": run_random_forest,
        "PLS-DA": run_pls_da
    }

    for model_type, model_func in models_config.items():
        model_results_list = []
        for strategy_name, folds_list in split_results_log.items():
            for i, (X_train, X_test, y_train, y_test) in enumerate(folds_list):
                if "Duplex" in strategy_name:
                    split_identifier = strategy_name
                else:
                    split_identifier = f"{strategy_name}_Fold{i + 1}"

                if model_type == "SVM":
                    y_pred = model_func(X_train, y_train, X_test)
                elif model_type in ["Logistic_Regression", "Random_Forest"]:
                    y_pred, _ = model_func(X_train, y_train, X_test)
                elif model_type == "PLS-DA":
                    y_pred, _, _ = model_func(X_train, y_train, X_test)

                metrics = compute_classification_metrics(
                    y_true=y_test,
                    y_pred=y_pred,
                    pos_label='CHD',
                    neg_label='CTRL',
                    model_name=split_identifier
                )

                model_results_list.append(metrics)

        if model_results_list:
            df_model_res = pd.DataFrame(model_results_list)
            filename = f"{model_type}_Performance.csv"
            save_path = os.path.join(training_dirs[2], filename)
            df_model_res.to_csv(save_path, index=False)

    '''


    #z_score_plot(df_neg_raw, "Z-score negative Raw", plots_dir)
    #z_score_plot(df_pos_raw, "Z-score positive Raw", plots_dir)

    #internal_variability(df_neg_raw, "Internal variability negative Raw", plots_dir)
    #internal_variability(df_pos_raw, "Internal variability positive Raw", plots_dir)

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