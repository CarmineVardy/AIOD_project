import pandas as pd
import os


def generate_dataset_report(df, dataset_name, output_path):
    """
    Analizza il dataset e salva un report strutturato in formato CSV.
    Colonne: Metric | Value
    """

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Preparazione dei dati per il calcolo
    sample_names = df.index
    classes = df['Class']
    numeric_data = df.drop(columns=['Class'])
    feature_names = numeric_data.columns

    # --- LISTA PER RACCOGLIERE I DATI ---
    report_data = []

    # 1. OVERVIEW
    n_features = len(feature_names)
    n_samples_total = len(sample_names)

    report_data.append({'Metric': 'Dataset Name', 'Value': dataset_name})
    report_data.append({'Metric': 'Total Features (Metabolites)', 'Value': n_features})
    report_data.append({'Metric': 'Total Samples', 'Value': n_samples_total})

    # 2. CLASS DISTRIBUTION (Totali grezzi)
    class_counts = classes.value_counts()
    for cls, count in class_counts.items():
        report_data.append({'Metric': f'{cls} Total', 'Value': count})

    # 3. NOMENCLATURE ANALYSIS
    samples_str = sample_names.astype(str)
    with_suffix_00 = [s for s in samples_str if '_00' in s]
    with_suffix_01 = [s for s in samples_str if '_01' in s]
    without_suffix = [s for s in samples_str if '_00' not in s and '_01' not in s]
    total_biological = len(with_suffix_00) + len(without_suffix)

    report_data.append({'Metric': "Samples with '_00'", 'Value': len(with_suffix_00)})
    report_data.append({'Metric': "Samples with '_01' (Tech Rep)", 'Value': len(with_suffix_01)})
    report_data.append({'Metric': "Samples without suffix", 'Value': len(without_suffix)})
    report_data.append({'Metric': "Est. Unique Biological Samples", 'Value': total_biological})

    # 4. DETAILED BREAKDOWN (Biological vs Technical)
    ctrl_primary = 0;
    ctrl_replicate = 0
    chd_primary = 0;
    chd_replicate = 0
    qc_count = 0

    for name, label in zip(samples_str, classes):
        if label == 'QC':
            qc_count += 1
            continue

        is_replicate = '_01' in name
        if label == 'CTRL':
            if is_replicate:
                ctrl_replicate += 1
            else:
                ctrl_primary += 1
        elif label == 'CHD':
            if is_replicate:
                chd_replicate += 1
            else:
                chd_primary += 1

    # Aggiungiamo i dettagli al report
    report_data.append({'Metric': 'CTRL - Biological', 'Value': ctrl_primary})
    report_data.append({'Metric': 'CTRL - Technical Rep.', 'Value': ctrl_replicate})
    report_data.append({'Metric': 'CHD - Biological', 'Value': chd_primary})
    report_data.append({'Metric': 'CHD - Technical Rep.', 'Value': chd_replicate})
    report_data.append({'Metric': 'QC - Total', 'Value': qc_count})

    # 5. DATA INTEGRITY
    total_cells = n_features * n_samples_total
    missing_values = numeric_data.isnull().sum().sum()
    missing_percentage = (missing_values / total_cells) * 100
    has_negative = (numeric_data < 0).any().any()

    report_data.append({'Metric': 'Missing Values (Count)', 'Value': missing_values})
    report_data.append({'Metric': 'Missing Values (%)', 'Value': f"{missing_percentage:.2f}%"})
    report_data.append({'Metric': 'Negative Values Present', 'Value': "YES" if has_negative else "NO"})

    # --- CREAZIONE DATAFRAME E SALVATAGGIO ---
    df_report = pd.DataFrame(report_data)

    # Salvataggio CSV
    try:
        df_report.to_csv(output_path, index=False)
        print(f"✅ CSV Report saved successfully in: {output_path}")
    except Exception as e:
        print(f"❌ Error saving CSV report: {e}")