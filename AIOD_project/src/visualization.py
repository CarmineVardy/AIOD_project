import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def run_pca_and_save_plots(df, dataset_label, output_folder):
    """
    Esegue la PCA e salva 3 grafici separati in formato PDF (Vettoriale).
    """

    # Crea la cartella se non esiste
    os.makedirs(output_folder, exist_ok=True)

    print(f"   -> Generazione grafici PCA (PDF) per: {dataset_label}...")

    # --- 1. CALCOLO PCA ---
    X = df.drop(columns=['Class'])
    y = df['Class']

    # Autoscaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA Fit
    n_comps_table = 5
    pca = PCA(n_components=10)
    scores = pca.fit_transform(X_scaled)

    # Calcolo Varianza
    explained_var = pca.explained_variance_ratio_ * 100
    cum_var = np.cumsum(explained_var)

    # DataFrame per gli Scores
    scores_df = pd.DataFrame(scores[:, :2], columns=['PC1', 'PC2'], index=df.index)
    scores_df['Class'] = y

    # Loadings
    loadings = pca.components_.T

    # Palette
    custom_palette = {'CTRL': '#0072B2', 'CHD': '#D55E00', 'QC': '#009E73'}
    custom_markers = {'CTRL': 'o', 'CHD': 's', 'QC': 'D'}

    # ==========================================
    #  PLOT 1: SCORES PLOT (PDF)
    # ==========================================
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=scores_df, x='PC1', y='PC2',
        hue='Class', style='Class',
        palette=custom_palette, markers=custom_markers,
        s=120, alpha=0.9, edgecolor='black'
    )

    plt.axhline(0, color='grey', linestyle='--', linewidth=1)
    plt.axvline(0, color='grey', linestyle='--', linewidth=1)

    plt.xlabel(f'PC1 ({explained_var[0]:.2f}%)', fontsize=12, fontweight='bold')
    plt.ylabel(f'PC2 ({explained_var[1]:.2f}%)', fontsize=12, fontweight='bold')
    plt.title(f'PCA Scores Plot: {dataset_label}', fontsize=14, fontweight='bold')

    # Legenda esterna
    plt.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.4)

    # Salvataggio in PDF
    # bbox_inches='tight' è CRUCIALE: evita che la legenda esterna venga tagliata
    filename_scores = f"PCA_Scores_{dataset_label}.pdf"
    plt.savefig(os.path.join(output_folder, filename_scores), format='pdf', bbox_inches='tight')
    plt.close()

    # ==========================================
    #  PLOT 2: LOADINGS PLOT (PDF)
    # ==========================================
    plt.figure(figsize=(10, 8))
    plt.scatter(
        loadings[:, 0], loadings[:, 1],
        alpha=0.7, color='#000080', s=30, edgecolor='white', linewidth=0.5
    )
    plt.axhline(0, color='black', linestyle='-', linewidth=1)
    plt.axvline(0, color='black', linestyle='-', linewidth=1)
    plt.xlabel(f'PC1 ({explained_var[0]:.2f}%)', fontsize=12, fontweight='bold')
    plt.ylabel(f'PC2 ({explained_var[1]:.2f}%)', fontsize=12, fontweight='bold')
    plt.title(f'PCA Loadings Plot: {dataset_label}', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.4)

    filename_loadings = f"PCA_Loadings_{dataset_label}.pdf"
    plt.savefig(os.path.join(output_folder, filename_loadings), format='pdf', bbox_inches='tight')
    plt.close()

    # ==========================================
    #  PLOT 3: VARIANCE TABLE (PDF)
    # ==========================================
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    ax.set_title(f'Explained Variance: {dataset_label}', fontsize=14, fontweight='bold')

    cell_text = []
    for i in range(n_comps_table):
        row = [f"PC{i + 1}", f"{explained_var[i]:.2f}%", f"{cum_var[i]:.2f}%"]
        cell_text.append(row)

    col_labels = ["Component", "Individual Variance", "Cumulative Variance"]

    the_table = ax.table(
        cellText=cell_text, colLabels=col_labels,
        loc='center', cellLoc='center', colColours=['#f2f2f2'] * 3
    )
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)
    the_table.scale(1, 1.5)

    filename_table = f"PCA_Variance_{dataset_label}.pdf"
    plt.savefig(os.path.join(output_folder, filename_table), format='pdf', bbox_inches='tight')
    plt.close()

    print(f"      ✅ Grafici PDF salvati in: {output_folder}")