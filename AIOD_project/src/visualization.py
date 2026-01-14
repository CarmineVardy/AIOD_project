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



def get_sample_names(df):
    """
    Extracts the first column of the dataframe to be used as sample identifiers.
    Assumes the first column contains the names/IDs.
    """
    # Reset index if the names are currently in the index
    if df.index.name is not None or not isinstance(df.index, pd.RangeIndex):
        return df.index.tolist()
    
    # Otherwise, return the first column
    return df.iloc[:, 0].tolist()

def biplot(df, dataset_label, output_folder):
    """
    Generates a PCA Biplot (Scores + Loadings) for the given dataframe.
    """
    # 1. Pre-processing: Separate Numeric Data from Metadata
    # The error 'could not convert string to float' happens because we are trying to scale text columns.
    df_numeric = df.select_dtypes(include=[np.number])
    
    # If the dataframe is empty after filtering, we cannot proceed
    if df_numeric.empty:
        print(f"Error: No numeric data found in {dataset_label} for PCA.")
        return

    # Extract sample names for grouping
    sample_names = get_sample_names(df)

    # Standardize the numeric data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_numeric)

    # PCA
    pca = PCA(n_components=5)
    scores = pca.fit_transform(X_scaled)
    loadings = pca.components_.T
    exp_var = pca.explained_variance_ratio_ * 100

    # 2. Setup Plot
    fig = plt.figure(figsize=(12, 10))
    scale_factor = np.max(np.abs(scores[:, :2])) * 0.8
    
    # Assign Groups based on Sample Names
    groups = []
    for name in sample_names:
        name_str = str(name).upper()
        if 'QC' in name_str:
            groups.append('QC')
        elif 'CTRL' in name_str:
            groups.append('Control')
        else:
            groups.append('Case')
    groups = np.array(groups)
    unique_groups = ['QC', 'Control', 'Case']

    group_colors = {
        'QC': '#FC8961',
        'Control': '#B73779',
        'Case': '#51127C'
    }
    
    # 3. Plot Scores (Samples)
    for group in unique_groups:
        mask = groups == group
        if np.any(mask):
            plt.scatter(scores[mask, 0], scores[mask, 1],
                        c=group_colors.get(group, 'grey'),
                        label=group, 
                        alpha=0.6, 
                        s=60, edgecolors='w')

    # 4. Plot Top 10 Contributors (Loadings)
    # Calculate magnitude of loadings
    magnitude = np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2)
    top_indices = np.argsort(magnitude)[-10:]
    
    # Get Feature Names (Columns of numeric df)
    metabolite_names = df_numeric.columns
    
    cmap = plt.get_cmap("magma")
    colors_contributors = [cmap(i) for i in np.linspace(0, 1, len(top_indices))]
    
    for idx, i in enumerate(top_indices):
        x_end = loadings[i, 0] * scale_factor
        y_end = loadings[i, 1] * scale_factor
        c = colors_contributors[idx]
        feat_name = metabolite_names[i]
        
        # Vector line
        plt.plot([0, x_end], [0, y_end], color=c, alpha=0.8, linewidth=1.5)
        # Endpoint
        plt.scatter(x_end, y_end, color=c, s=40, zorder=10)
        # Label
        plt.text(x_end * 1.1, y_end * 1.1, feat_name,
                 color=c, fontsize=8, fontweight='bold', ha='center')
    
    # Decorations
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='grey', linestyle='--', linewidth=0.8)
    plt.xlabel(f'PC1 ({exp_var[0]:.2f}%)')
    plt.ylabel(f'PC2 ({exp_var[1]:.2f}%)')
    plt.title(f'Biplot: Samples & Contributors - {dataset_label}')
    plt.legend(loc='upper right', framealpha=0.9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save/Show
    # if output_folder:
    #     plt.savefig(f"{output_folder}/Biplot_{dataset_label}.png", dpi=300)
    plt.show()

def z_score_plot(df, dataset_label, output_folder, samples_per_group=25):
    """
    Calculates Z-scores for a subset of samples (Control vs Case) and plots them.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The TRANSPOSED dataframe (Rows=Samples, Columns=Features).
    dataset_label : str
        Label for the plot title.
    output_folder : str
        Path to save the plot (optional).
    samples_per_group : int, default=25
        Number of samples to select from each group (Control and Case).
    """
    # 1. Prepare Data: Ensure we are working with numeric features for calculation
    # We assume 'df' is already transposed, so index contains sample names.
    df_numeric = df.select_dtypes(include=[np.number])
    
    # 2. Identify Groups based on Index (Sample Names)
    # Filter out QCs first
    mask_qc = df.index.str.contains('QC', case=False, na=False)
    df_no_qc = df_numeric[~mask_qc]
    
    # Identify Controls (containing 'CTRL') and Cases (the rest)
    mask_ctrl = df_no_qc.index.str.contains('CTRL', case=False, na=False)
    
    df_ctrl = df_no_qc[mask_ctrl]
    df_case = df_no_qc[~mask_ctrl]
    
    # 3. Select Subset (x samples per group)
    # We take the first 'samples_per_group' available. 
    # If fewer are available, we take all of them.
    subset_ctrl = df_ctrl.iloc[:samples_per_group]
    subset_case = df_case.iloc[:samples_per_group]
    
    # Combine the subsets
    df_subset = pd.concat([subset_ctrl, subset_case])
    
    if df_subset.empty:
        print("Error: No samples found for Z-score plotting after filtering.")
        return

    # 4. Calculate Z-Scores
    # Calculate Sum of intensities per sample (Total Ion Current proxy)
    sample_sums = df_subset.sum(axis=1)
    
    # Calculate Z-Score relative to this subset's mean/std
    z_score_data = (sample_sums - sample_sums.mean()) / sample_sums.std()
    
    # 5. Assign Colors using Magma Palette
    # We pick two distinct colors from the Magma colormap
    # 0.2 is dark purple/black, 0.7 is reddish/orange
    cmap = plt.get_cmap("magma")
    color_ctrl = cmap(0.2) 
    color_case = cmap(0.7)
    
    # Create a color list corresponding to the rows in df_subset
    # Since we concatenated [ctrl, case], the first N are ctrl, the rest are case.
    colors = [color_ctrl] * len(subset_ctrl) + [color_case] * len(subset_case)
    
    # 6. Plotting
    plt.figure(figsize=(15, 6))
    
    bars = plt.bar(range(len(z_score_data)), z_score_data, color=colors, alpha=0.9)
    
    plt.title(f'Z-Scores of Sample Intensities (Subset: {samples_per_group}/group) - {dataset_label}', fontsize=14)
    plt.ylabel('Z-Score')
    plt.axhline(y=0, color='grey', linestyle='-', linewidth=0.8)
    
    # Set X-ticks with sample names
    plt.xticks(range(len(z_score_data)), df_subset.index, rotation=90, fontsize=8)
    
    # Add threshold lines
    plt.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='Threshold (+/- 2)')
    plt.axhline(y=-2, color='red', linestyle='--', alpha=0.5)
    
    # 7. Create Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=color_ctrl, lw=4, label='Control'),
        Line2D([0], [0], color=color_case, lw=4, label='Case'),
        Line2D([0], [0], color='red', linestyle='--', label='Outlier Threshold')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.grid(True, axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()

    '''
    # Save/Show
    if output_folder:
        import os
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        plt.savefig(os.path.join(output_folder, f"Z_Score_Subset_{dataset_label}.png"), dpi=300)
    '''
    plt.show()


def internal_variability(df, dataset_label, output_folder, samples_per_group=25):
    """
    Generates a boxplot representing the distribution of intensities for each sample.
    Filters for a subset of samples (Control vs Case) similar to the Z-score plot.
    """
    # 1. Filter Data for Subset (Same logic as Z-score plot)
    # Ensure we are working with numeric features for plotting
    df_numeric_full = df.select_dtypes(include=[np.number])
    
    # Filter out QCs
    mask_qc = df.index.str.contains('QC', case=False, na=False)
    df_no_qc = df_numeric_full[~mask_qc]
    
    # Identify Controls and Cases
    mask_ctrl = df_no_qc.index.str.contains('CTRL', case=False, na=False)
    df_ctrl = df_no_qc[mask_ctrl]
    df_case = df_no_qc[~mask_ctrl]
    
    # Select Subset
    subset_ctrl = df_ctrl.iloc[:samples_per_group]
    subset_case = df_case.iloc[:samples_per_group]
    
    # Combine the subsets
    df_subset = pd.concat([subset_ctrl, subset_case])
    
    if df_subset.empty:
        print("Error: No samples found for Internal Variability plotting after filtering.")
        return

    # 2. Prepare Data for Boxplot
    # Transpose so columns are samples (Seaborn interprets columns as x-axis categories)
    df_plot = df_subset.T 
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Generate Boxplot
    sns.boxplot(data=df_plot, palette="magma", ax=ax, showfliers=False, linewidth=0.8)
    
    # Aesthetics
    ax.set_title(f'Internal Variability (Subset: {samples_per_group}/group) - {dataset_label}', fontsize=14, fontweight='bold')
    ax.set_xlabel("Samples", fontsize=12)
    ax.set_ylabel("Log Intensity / Abundance", fontsize=12)
    
    # Manage x-tick density using labels from the helper function
    sample_names = get_sample_names(df_subset)
    
    if df_plot.shape[1] > 60:
        ax.set_xticks([]) # Hide labels if too crowded
    else:
        # Set ticks manually to ensure alignment with sample names
        ax.set_xticks(range(len(sample_names)))
        ax.set_xticklabels(sample_names, rotation=90, fontsize=8)
        
    plt.tight_layout()
    
    # Save/Show logic can be added here if needed, consistent with z_score_plot
    if output_folder:
        import os
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        plt.savefig(os.path.join(output_folder, f"Internal_Variability_Subset_{dataset_label}.png"), dpi=300)

    plt.show()
    
    