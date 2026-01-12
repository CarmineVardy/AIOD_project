import pandas as pd
import os

def load_raw_dataset(file_path):
    """Carica il CSV grezzo originale."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ERRORE: File raw non trovato: {file_path}")
    return pd.read_csv(file_path)

def load_transposed_dataset(file_path):
    """
    Carica il CSV invertito.
    Usa index_col=0 per impostare correttamente i nomi dei campioni come indice.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ERRORE: Dataset non trovato: {file_path}. Esegui la funzione di generazione.")

    return pd.read_csv(file_path, index_col=0)

def reshape_dataset(df_raw):
    """Inverte il dataset (Transpose) e sistema le classi."""
    samples = df_raw.columns[1:]
    classes = df_raw.iloc[0, 1:].values
    metabolites = df_raw.iloc[1:, 0].values

    # Transpose
    data_matrix = df_raw.iloc[1:, 1:].values.T

    df_reshaped = pd.DataFrame(data_matrix, index=samples, columns=metabolites)
    df_reshaped = df_reshaped.apply(pd.to_numeric, errors='coerce')
    df_reshaped.insert(0, 'Class', classes)

    return df_reshaped


def remove_qc_and_technical_replicates(df):
    """
    Rimuove i campioni di classe 'QC' e i duplicati tecnici (quelli che contengono '_01').
    Restituisce il DataFrame pulito.
    """
    # 1. Rimuovi i QC
    df_clean = df[df['Class'] != 'QC'].copy()

    # 2. Rimuovi i duplicati tecnici (quelli che CONTENGONO '_01')
    # Usiamo str.contains invece di endswith per intercettarlo ovunque nel nome
    # regex=False assicura che cerchi esattamente la stringa "_01" senza interpretarla come codice
    df_clean = df_clean[~df_clean.index.str.contains('_01', regex=False)]

    return df_clean