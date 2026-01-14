import os
import pandas as pd

from src.data_loader import *
from src.analysis import *
from src.dataFusion import *
from src.anomalyDetection import *

from src.datasetSplitting import *

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

def generate_transposed_datasets():
    """Genera i file CSV trasposti dai Raw originali."""
    datasets = [
        (NEG_RAW_PATH, NEG_TRANSPOSED_PATH, "Negative"),
        (POS_RAW_PATH, POS_TRANSPOSED_PATH, "Positive")
    ]
    os.makedirs(TRANSPOSED_DIR, exist_ok=True)
    for raw, trans, label in datasets:
        if os.path.exists(raw):
            df = load_raw_dataset(raw)
            df_t = reshape_dataset(df)
            df_t.to_csv(trans, index=True)
            print(f"Saved Transposed {label}")
        else:
            print(f"Raw {label} mancante.")

def generate_cleaned_datasets():
    """Carica i file Transposed, rimuove QC e Replicati tecnici, e salva i file Cleaned."""
    os.makedirs(CLEANED_DIR, exist_ok=True)
    datasets_to_process = [
        (NEG_TRANSPOSED_PATH, NEG_CLEANED_PATH, "Negative"),
        (POS_TRANSPOSED_PATH, POS_CLEANED_PATH, "Positive")
    ]
    for input_path, output_path, label in datasets_to_process:
        if os.path.exists(input_path):
            df_raw = load_transposed_dataset(input_path)
            df_clean = remove_qc_and_technical_replicates(df_raw)
            df_clean.to_csv(output_path, index=True)
        else:
            print(f"Impossibile generare Cleaned {label}: File Transposed mancante ({input_path})")


# ==============================================================================
#  ANALYSIS STEPS (LOGIC)
# ==============================================================================
def step1_quality_assessment(df_neg, df_pos, output_dir):
    """
    FASE 1: Valutazione Qualità Dati (QC, Replicati, Raw PCA).
    Accetta dataframe generici (in questo caso ci aspettiamo quelli Raw con QC).
    """
    print("\n--- AVVIO STEP 1: QUALITY ASSESSMENT ---")

    analysis_dir = os.path.join(output_dir, "dataset_analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    report_neg_path = os.path.join(analysis_dir, 'Preliminary_Analysis_Neg.csv')
    report_pos_path = os.path.join(analysis_dir, 'Preliminary_Analysis_Pos.csv')

    generate_dataset_report(df_neg, "ESI- Negative (Raw)", report_neg_path)
    generate_dataset_report(df_pos, "ESI+ Positive (Raw)", report_pos_path)

    plots_dir = os.path.join(output_dir, "plots")

    pca_neg_dir = os.path.join(plots_dir, "pca", "negative")
    pca_pos_dir = os.path.join(plots_dir, "pca", "positive")

    run_pca_and_save_plots(df_neg, "ESI_Negative_Raw", pca_neg_dir)
    run_pca_and_save_plots(df_pos, "ESI_Positive_Raw", pca_pos_dir)

    print("✅ Step 1 Completato.")


def step2_preprocessing(df_neg, df_pos, output_dir):
    """
    FASE 2: Preprocessing (Normalizzazione, Trasformazione Log, Scaling).
    Accetta dataframe generici (in questo caso ci aspettiamo quelli Cleaned senza QC e Duplicati).
    """
    print("\n--- AVVIO STEP 2: DATA PREPROCESSING ---")

    analysis_dir = os.path.join(output_dir, "dataset_analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    report_neg_path = os.path.join(analysis_dir, 'Preliminary_Analysis_Neg.csv')
    report_pos_path = os.path.join(analysis_dir, 'Preliminary_Analysis_Pos.csv')

    generate_dataset_report(df_neg, "ESI- Negative (Raw)", report_neg_path)
    generate_dataset_report(df_pos, "ESI+ Positive (Raw)", report_pos_path)


def main():

    #generate_transposed_datasets()
    #generate_cleaned_datasets()

    '''
    #STEP 1
    df_neg_raw = load_transposed_dataset(NEG_TRANSPOSED_PATH)
    df_pos_raw = load_transposed_dataset(POS_TRANSPOSED_PATH)
    os.makedirs(STEP1_DIR, exist_ok=True)
    step1_quality_assessment(
        df_neg_raw,
        df_pos_raw,
        STEP1_DIR
    )
    
    '''
    # STEP 2
    df_neg_clean = load_transposed_dataset(NEG_CLEANED_PATH)
    df_pos_clean = load_transposed_dataset(POS_CLEANED_PATH)
    os.makedirs(STEP2_DIR, exist_ok=True)
    step2_preprocessing(
        df_neg_clean,
        df_pos_clean,
        STEP2_DIR
    )
    

# ==============================================================================
#  DATASET MERGING
# ==============================================================================

    df_list = []

    df_list.append(df_neg_clean)
    df_list.append(df_pos_clean)

    dFusion = DataFusion(df_list)
    df_low_level_merged = dFusion.low_level_fusion()
    #df_qc_merged = dFusion.qc_based_fusion()

    #print(df_low_level_merged)
    #print(df_qc_merged)


    
    plots_dir = os.path.join(STEP2_DIR, "plots")

    biplot_neg_dir = os.path.join(plots_dir, "biplot", "negative")
    biplot_pos_dir = os.path.join(plots_dir, "bilot", "positive")

    biplot(df_low_level_merged, "Negative cleaned", biplot_neg_dir)
    biplot(df_pos_clean, "Positive cleaned", biplot_pos_dir)


    anomalyDete = AnomalyDetector()
    #z_scores_neg, std_dev_neg = anomalyDete.calculate_z_scores(df_neg_clean)

    #print(f"\nSample Std Dev: {std_dev_neg:.3f}")
    z_score_plot(df_neg_clean, "Negative cleaned", biplot_neg_dir)


    internal_variability(df_neg_clean, "Negative cleaned", biplot_neg_dir)


# ==============================================================================
#  DATASET SPLITTING
# ==============================================================================

    X = df_neg_clean.select_dtypes(include=[np.number])

    # Extract y (Labels) and Groups (Patient/Biological Source)
    # We derive these from the sample names (index or first column)
    sample_names = get_sample_names(df_neg_clean)
    classes = []
    groups = []

    for name in sample_names:
        s_name = str(name).upper()
        
        # Create Labels: Control vs Case
        if 'CTRL' in s_name:
            classes.append('Control')
        else:
            classes.append('Case')
        
        # Create Groups: Biological Replicates
        # Heuristic: We assume the part before the first underscore is the Patient ID
        # e.g., "CTRL01_00" -> Group "CTRL01". This ensures 00, 01, etc., stay together.
        groups.append(s_name.split('_')[0])

    # 3. Construct the Combined DataFrame for the Class
    # The new class requires the target and group to be columns inside the dataframe
    df_for_splitting = df_neg_clean.copy()
    df_for_splitting['Target_Class'] = classes
    df_for_splitting['Bio_Group'] = groups

    # 4. Instantiate the Splitting Class
    # We pass the column names for target and group
    splitter = DatasetSplitting(df_for_splitting, target_col='Target_Class', group_col='Bio_Group')

    # --- EXECUTE SPLIT TESTS ---
    
    # A. Train/Test Split
    print("\n--- 1. Running Train/Test Split ---")
    # This method in the new class returns DataFrames/Arrays directly
    X_train, X_test, y_train, y_test = splitter.split_train_test(test_size=0.3)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train Class Dist: {np.unique(y_train, return_counts=True)}")

    # B. Stratified K-Fold (Group-aware)
    print("\n--- 2. Running Stratified K-Fold ---")
    # This method in the new class returns a generator of indices
    kfold_gen = splitter.stratified_k_fold(n_splits=5)
    train_idx, test_idx = next(kfold_gen) # Get first fold
    print(f"Fold 1 - Train indices: {len(train_idx)}, Test indices: {len(test_idx)}")

    # C. Leave-One-Out (Group-aware -> Leave-One-Group-Out)
    print("\n--- 3. Running Leave-One-Out (LOGO) ---")
    loo_gen = splitter.leave_one_out()
    # Consume the generator to ensure it runs and updates the history
    splits_loo = list(loo_gen)
    print(f"Total LOO Splits generated: {len(splits_loo)}")

    # --- STEP 4: Visualization & Benchmarking ---

    # D. Visualize the splits
    # This generates the plot using the history stored in the class
    print("\n--- 4. Visualizing Splits ---")
    splitter.plot_splits_visualization()

    # E. Benchmark Model Performance
    # This compares accuracy across the different splitting methods
    print("\n--- 5. Benchmarking ---")
    benchmark_results = splitter.benchmark_methods()
    print("\nBenchmark Summary:")
    print(benchmark_results)


if __name__ == "__main__":
    main()