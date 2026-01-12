import os
import pandas as pd

from src.data_loader import *
from src.analysis import *
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
            print(f"    ✅ Saved Transposed {label}")
        else:
            print(f"    ❌ Raw {label} mancante.")

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
            print(f"    ❌ Impossibile generare Cleaned {label}: File Transposed mancante ({input_path})")


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

if __name__ == "__main__":
    main()