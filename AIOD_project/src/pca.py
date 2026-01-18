import os

import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import f, chi2

from src.config_visualization import *


def perform_pca(df, n_components=None, scaling='autoscaling'):
    """
    Performs Principal Component Analysis (PCA) on the provided dataset.

    THEORY & IMPLEMENTATION DETAILS:
    --------------------------------
    1. Preprocessing:
       - PCA is scale-dependent.
       - Options provided: 'autoscaling' (mean=0, std=1) or 'pareto' (mean=0, std=sqrt(std)).
       - User must ensure data is cleaned (no NaNs) before calling this.
    2. Calculation:
       - Uses SVD decomposition via sklearn.
    3. Output Organization:
       - Returns a dictionary containing Scores (Samples), Loadings (Features),
         and Variance stats, all labeled with proper indices.

    Args:
        df (pd.DataFrame): Input data (Samples x Features).
        n_components (int): Number of components to keep. If None, keeps all.
        scaling (str): 'autoscaling', 'pareto', or None.
                       - 'autoscaling': Recommended for general LC-MS.
                       - 'pareto': Recommended if noise is high (reduces noise amplification).
                       - None: Assumes data is already scaled externally.

    Returns:
        dict: A dictionary containing:
            - 'model': The trained PCA sklearn object.
            - 'scores': pd.DataFrame of Score values (Samples x PCs).
            - 'loadings': pd.DataFrame of Loading values (Features x PCs).
            - 'explained_variance': Array of variance ratio per PC.
            - 'cumulative_variance': Array of cumulative variance.
    """
    # Separate numeric data for PCA calculation
    df = df.drop(columns=['Class'])

    # 1. Scaling / Preprocessing
    data_mat = df.values

    if scaling == 'autoscaling':
        # Mean=0, Std=1
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_mat)
    elif scaling == 'pareto':
        # Mean Centering
        data_centered = data_mat - np.mean(data_mat, axis=0)
        # Pareto: Divide by sqrt(STD)
        data_scaled = data_centered / np.sqrt(np.std(data_mat, axis=0))
    else:
        # Assume external scaling or just Mean Centering (default in PCA)
        data_scaled = data_mat

    # 2. PCA Execution
    # If n_components is None, sklearn computes all min(n_samples, n_features)
    pca = PCA(n_components=n_components)
    scores_data = pca.fit_transform(data_scaled)

    # 3. Formatting Outputs
    # Create column names [PC1, PC2, ..., PCn]
    pc_labels = [f"PC{i + 1}" for i in range(scores_data.shape[1])]

    # Scores DataFrame (Rows=Samples)
    scores_df = pd.DataFrame(data=scores_data, index=df.index, columns=pc_labels)

    # Loadings DataFrame (Rows=Features)
    # sklearn components_ is (n_components, n_features), so we transpose
    loadings_df = pd.DataFrame(data=pca.components_.T, index=df.columns, columns=pc_labels)

    results = {
        'model': pca,
        'scores': scores_df,
        'loadings': loadings_df,
        'explained_variance': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
        'data_scaled': data_scaled
    }

    return results

def plot_pca_scree(pca_results, output_dir, file_name="pca_scree_plot", threshold=0.9):
    """
    Generates a Scree Plot (Elbow Method) to help select the optimal number of components.

    Displays both individual Explained Variance (Bars) and Cumulative Variance (Line).
    """
    os.makedirs(output_dir, exist_ok=True)

    var_ratio = pca_results['explained_variance']
    cum_var = pca_results['cumulative_variance']
    n_pcs = len(var_ratio)

    # Limit to first 20 PCs for readability if many dimensions exist
    n_plot = min(20, n_pcs)
    x_range = np.arange(1, n_plot + 1)

    fig, ax1 = plt.subplots()

    # Bar Chart (Individual Variance)
    ax1.bar(x_range, var_ratio[:n_plot] * 100, color=DISCRETE_COLORS[0], alpha=0.7, label='Individual Variance')
    ax1.set_xlabel('Principal Components')
    ax1.set_ylabel('Explained Variance (%)', color=DISCRETE_COLORS[0])
    ax1.tick_params(axis='y', labelcolor=DISCRETE_COLORS[0])
    ax1.set_xticks(x_range)

    # Line Chart (Cumulative Variance)
    ax2 = ax1.twinx()
    ax2.plot(x_range, cum_var[:n_plot] * 100, color=DISCRETE_COLORS[4], marker='o', linewidth=2,
             label='Cumulative Variance')
    ax2.set_ylabel('Cumulative Variance (%)', color=DISCRETE_COLORS[4])
    ax2.tick_params(axis='y', labelcolor=DISCRETE_COLORS[4])

    # Threshold Line (e.g., 90%)
    ax2.axhline(y=threshold * 100, color='grey', linestyle='--', alpha=0.5)
    ax2.text(n_plot, threshold * 100 + 1, f'{int(threshold * 100)}% Threshold', color='grey', ha='right')

    #Legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')

    plt.title('PCA Scree Plot (Explained Variance)', fontweight='bold')

    save_path = os.path.join(output_dir, f"{file_name}.{SAVE_FORMAT}")
    plt.savefig(save_path, format=SAVE_FORMAT, bbox_inches='tight')
    plt.close(fig)
    print(f"Generating PCA Scree Plot saved to {save_path}")


def plot_sum_pca_scree_contribution(pca_results, output_dir, file_name="sum_pca_scree_blocks_grouped", threshold=0.9):
    """
    Generates a Grouped (Side-by-Side) Scree Plot for SUM-PCA.

    VISUALIZATION:
    - Instead of stacking, it places the bar for Block 1 and Block 2 side-by-side
      for each Principal Component.
    - This allows for a direct comparison of which block contributes more to a specific PC.

    THEORY:
    - The contribution of a block to a specific PC is proportional to the sum of
      squared loadings of the features belonging to that block.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract data from results dictionary
    var_ratio = pca_results['explained_variance']
    cum_var = pca_results['cumulative_variance']
    loadings_df = pca_results['loadings']
    n_pcs = len(var_ratio)

    # Limit visualization to first 20 PCs (or less if fewer exist)
    n_plot = min(20, n_pcs)

    # X-axis locations for the groups
    x = np.arange(n_plot)
    width = 0.35  # Width of the bars

    # --- 1. CALCULATE BLOCK CONTRIBUTIONS PER PC ---
    var_b1 = []  # To store variance explained by Block 1 (Neg)
    var_b2 = []  # To store variance explained by Block 2 (Pos)

    for i in range(n_plot):
        pc_name = f"PC{i + 1}"
        pc_loadings = loadings_df[pc_name]

        # Calculate Sum of Squared Loadings for each block
        # We assume features are suffixed with '_Block1' and '_Block2'
        sq_load_b1 = pc_loadings[pc_loadings.index.str.contains("_Block1")].pow(2).sum()
        sq_load_b2 = pc_loadings[pc_loadings.index.str.contains("_Block2")].pow(2).sum()

        # Total squared loadings for this PC (should be approx 1.0)
        total_sq = sq_load_b1 + sq_load_b2

        # Calculate Proportions
        prop_b1 = sq_load_b1 / total_sq
        prop_b2 = sq_load_b2 / total_sq

        # Calculate actual Variance Explained % for each block
        real_var_b1 = prop_b1 * var_ratio[i] * 100
        real_var_b2 = prop_b2 * var_ratio[i] * 100

        var_b1.append(real_var_b1)
        var_b2.append(real_var_b2)

    # --- 2. PLOTTING ---
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Grouped Bars logic:
    # Block 1 bars are shifted to the left (x - width/2)
    # Block 2 bars are shifted to the right (x + width/2)
    rects1 = ax1.bar(x - width / 2, var_b1, width, label='Block 1 (Neg)', color=DISCRETE_COLORS[0], alpha=0.8)
    rects2 = ax1.bar(x + width / 2, var_b2, width, label='Block 2 (Pos)', color=DISCRETE_COLORS[1], alpha=0.8)

    # Labels and Titles
    ax1.set_xlabel('Principal Components (Super Scores)')
    ax1.set_ylabel('Explained Variance (%)')
    ax1.set_title('SUM-PCA: Variance Contribution per Block (Grouped)', fontweight='bold')

    # Set X-ticks to be in the center of the group
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"PC{i + 1}" for i in range(n_plot)])

    # --- 3. CUMULATIVE VARIANCE LINE (Secondary Axis) ---
    ax2 = ax1.twinx()
    ax2.plot(x, cum_var[:n_plot] * 100, color='black', marker='o', linewidth=2, linestyle='-',
             label='Cumulative Variance')
    ax2.set_ylabel('Cumulative Variance (%)')

    # Set y-limit for cumulative variance to 0-105% for better readability
    ax2.set_ylim(0, 105)

    # Threshold Line (e.g. 90%)
    ax2.axhline(y=threshold * 100, color='grey', linestyle='--', alpha=0.5)
    ax2.text(n_plot - 1, threshold * 100 + 2, f'{int(threshold * 100)}% Threshold', color='grey', ha='right')

    # --- 4. LEGEND ---
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    plt.tight_layout()

    # Save
    save_path = os.path.join(output_dir, f"{file_name}.{SAVE_FORMAT}")
    plt.savefig(save_path, format=SAVE_FORMAT, bbox_inches='tight')
    plt.close(fig)
    print(f"Generating SUM-PCA Grouped Scree Plot saved to {save_path}")

def plot_pca_scores(pca_results, df, output_dir, pc_x=1, pc_y=2, file_name="pca_score_plot", class_col='Class',
                    show_ellipse=True, is_sum_pca=False):
    """
    Generates the PCA Score Plot with Hotelling's T2 Confidence Ellipse.

    THEORY:
    - Visualizes samples in the reduced space (PCx vs PCy).
    - Ellipse: Represents the 95% Confidence Interval for the Normal Distribution of the model.
      Strictly calculated on the GLOBAL dataset (unsupervised), not per-class.
    - Outliers: Samples outside the ellipse are potential candidates based on T2 distance.
    """
    os.makedirs(output_dir, exist_ok=True)

    scores = pca_results['scores']
    var_ratio = pca_results['explained_variance']

    # Get PC labels and data
    pc_x_label = f"PC{pc_x}"
    pc_y_label = f"PC{pc_y}"

    x_data = scores[pc_x_label]
    y_data = scores[pc_y_label]

    fig, ax = plt.subplots()

    #Handling of class NotShow if we don't want to see class in plotting
    mask_hidden = df[class_col] == 'NotShow'
    mask_visible = ~mask_hidden

    if mask_hidden.any():
        ax.scatter(x_data[mask_hidden], y_data[mask_hidden],
                   alpha=0, s=100, edgecolor='none')

    visible_classes = df.loc[mask_visible, class_col].unique()
    current_palette = DISCRETE_COLORS[:len(visible_classes)]
    markers_dict = {cls: MARKERS[i % len(MARKERS)] for i, cls in enumerate(visible_classes)}

    # Scatter Plot
    sns.scatterplot(x=x_data[mask_visible], y=y_data[mask_visible],
                    hue=df.loc[mask_visible, class_col],
                    style=df.loc[mask_visible, class_col],
                    palette=current_palette, markers=markers_dict,
                    s=100, alpha=0.8, edgecolor='k', ax=ax)

    # --- HOTELLING'S T2 ELLIPSE CALCULATION ---
    # --- CLASS-SPECIFIC CONFIDENCE ELLIPSES ---
    if show_ellipse:
        # Calcoliamo un'ellisse per OGNI classe visibile
        for i, cls in enumerate(visible_classes):
            # Filtriamo i punti di questa specifica classe
            cls_mask = (df[class_col] == cls) & mask_visible
            x_cls = x_data[cls_mask]
            y_cls = y_data[cls_mask]

            # Servono almeno 3 punti per definire un'ellisse (varianza)
            if len(x_cls) > 2:
                # 1. Calcolo Covarianza e Media del gruppo
                cov = np.cov(x_cls, y_cls)
                mean_x = np.mean(x_cls)
                mean_y = np.mean(y_cls)

                # 2. Autovalori/Autovettori per determinare l'orientamento
                lambda_, v = np.linalg.eig(cov)
                # Ordiniamo per grandezza decrescente
                order = lambda_.argsort()[::-1]
                lambda_, v = lambda_[order], v[:, order]

                # Calcolo angolo di rotazione
                angle = np.degrees(np.arctan2(*v[:, 0][::-1]))

                # 3. Dimensioni per il 95% di confidenza (Chi-quadro 2 gradi libertà = 5.991)
                # width/height = 2 * sqrt(5.991 * autovalore)
                scale_factor = 2 * np.sqrt(5.991)
                width, height = scale_factor * np.sqrt(lambda_)

                # 4. Disegno Ellisse
                # Usiamo lo stesso colore dei punti (current_palette[i])
                color = current_palette[i]

                # Bordo tratteggiato
                ellipse_edge = Ellipse(xy=(mean_x, mean_y), width=width, height=height, angle=angle,
                                       facecolor='none', edgecolor=color, linestyle='--', linewidth=1.5,
                                       label=f'{cls} 95% CI')
                ax.add_patch(ellipse_edge)

    # Labels with Explained Variance
    var_x = var_ratio[pc_x - 1] * 100
    var_y = var_ratio[pc_y - 1] * 100

    ax.set_xlabel(f"{pc_x_label} ({var_x:.1f}%)", fontweight='bold')
    ax.set_ylabel(f"{pc_y_label} ({var_y:.1f}%)", fontweight='bold')
    if is_sum_pca:
        title_text = f"SUM-PCA Super Scores: PC{pc_x} vs PC{pc_y}"
    else:
        title_text = f"PCA Scores Plot: PC{pc_x} ({var_x:.1f}%) vs PC{pc_y} ({var_y:.1f}%)"

    ax.set_title(title_text, fontweight='bold')

    # Center lines
    ax.axhline(0, color='grey', linestyle=':', linewidth=0.8)
    ax.axvline(0, color='grey', linestyle=':', linewidth=0.8)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{file_name}_PC{pc_x}vsPC{pc_y}.{SAVE_FORMAT}")
    plt.savefig(save_path, format=SAVE_FORMAT, bbox_inches='tight')
    plt.close(fig)
    print(f"Generating PCA Score Plot ({pc_x}vs{pc_y}) saved to {save_path}")


def plot_pca_loadings(pca_results, output_dir, pc_x=1, pc_y=2, file_name="pca_loading_plot", top_n=20):
    """
    Generates the PCA Loading Plot to interpret variable contributions.

    Shows the weight of each metabolite on the selected PCs.
    Since we have many features, we use a Scatter representation (Simpler Biplot concept).
    """
    os.makedirs(output_dir, exist_ok=True)

    loadings = pca_results['loadings']
    pc_x_label = f"PC{pc_x}"
    pc_y_label = f"PC{pc_y}"

    # Calculate magnitude (distance from origin) to identify top contributors
    magnitude = np.sqrt(loadings[pc_x_label] ** 2 + loadings[pc_y_label] ** 2)
    loadings['magnitude'] = magnitude

    # Sort and take top N for labeling
    top_loadings = loadings.sort_values(by='magnitude', ascending=False).head(top_n)

    fig, ax = plt.subplots()

    # Plot all features
    # Usiamo un colore secondario della palette (es. ind. 3) per lo sfondo, con trasparenza
    ax.scatter(loadings[pc_x_label], loadings[pc_y_label], color=DISCRETE_COLORS[3], alpha=0.7, s=20, label='All Features')

    # Plot top features as highlighted points
    if top_n > 0:
        ax.scatter(top_loadings[pc_x_label], top_loadings[pc_y_label], color=DISCRETE_COLORS[9], s=50,
                   label=f'Top {top_n} Contributors')

    # Annotate Top N
    texts = []
    for feature, row in top_loadings.iterrows():
        texts.append(plt.text(row[pc_x_label], row[pc_y_label], feature, fontsize=8, color=DISCRETE_COLORS[9],
                              fontweight='bold'))

    # Center lines
    ax.axhline(0, color='k', linestyle='-', linewidth=0.8)
    ax.axvline(0, color='k', linestyle='-', linewidth=0.8)

    # Draw circle of correlation (optional visual aid, radius depends on scaling, usually 1 or max loading)
    # Here we just keep axes symmetric
    max_val = max(abs(loadings[pc_x_label].max()), abs(loadings[pc_y_label].max())) * 1.1
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)

    ax.set_xlabel(f"Loading {pc_x_label}", fontweight='bold')
    ax.set_ylabel(f"Loading {pc_y_label}", fontweight='bold')
    ax.set_title(f"PCA Loading Plot: {pc_x_label} vs {pc_y_label}", fontweight='bold')
    ax.legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{file_name}_PC{pc_x}vsPC{pc_y}.{SAVE_FORMAT}")
    plt.savefig(save_path, format=SAVE_FORMAT, bbox_inches='tight')
    plt.close(fig)
    print(f"Generating PCA Loading Plot saved to {save_path}")


def plot_sum_pca_loadings(pca_results, output_dir, pc_x=1, pc_y=2, file_name="sum_pca_loadings_blocks", top_n=20):
    """
    Generates a variant of the PCA Loading Plot specifically for SUM-PCA.

    DIFFERENCE:
    - Colors features based on their Source Block (e.g., Block1 vs Block2).
    - Allows visual assessment of which block contributes most to the Super Scores.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Lavoriamo su una copia per non modificare il dizionario originale
    loadings = pca_results['loadings'].copy()
    pc_x_label = f"PC{pc_x}"
    pc_y_label = f"PC{pc_y}"

    # --- 1. BLOCK IDENTIFICATION LOGIC ---
    # Creiamo una colonna 'Block' basandoci sul suffisso del nome della feature
    # (Assumiamo la convenzione usata in low_level_fusion: "_Block1", "_Block2")
    def identify_block(feature_name):
        if "_Block1" in feature_name:
            return "Block 1 (Neg)"
        elif "_Block2" in feature_name:
            return "Block 2 (Pos)"
        else:
            return "Unknown"

    loadings['Block'] = loadings.index.to_series().apply(identify_block)

    # --- 2. CALCULATE MAGNITUDE (for Top N labeling) ---
    magnitude = np.sqrt(loadings[pc_x_label] ** 2 + loadings[pc_y_label] ** 2)
    loadings['magnitude'] = magnitude

    # Identifichiamo i Top N globali (i più influenti indipendentemente dal blocco)
    top_loadings = loadings.sort_values(by='magnitude', ascending=False).head(top_n)

    # --- 3. PLOTTING ---
    fig, ax = plt.subplots(figsize=(10, 8))

    # Definizione Palette Dinamica per i Blocchi
    unique_blocks = sorted(loadings['Block'].unique())
    # Usiamo i primi colori della tua palette globale
    block_palette = DISCRETE_COLORS[:len(unique_blocks)]

    # Scatter Plot con 'hue' basato sul Blocco
    sns.scatterplot(
        data=loadings,
        x=pc_x_label,
        y=pc_y_label,
        hue='Block',  # Colora in base al blocco
        style='Block',  # Cambia anche la forma del punto per accessibilità
        palette=block_palette,
        alpha=0.7,
        s=40,  # Punti leggermente più grandi
        edgecolor='none',  # Rimuove bordo per pulizia
        ax=ax
    )

    # --- 4. ANNOTATION (Solo i Top N Globali) ---
    # Annotiamo i punti più distanti dall'origine, colorando il testo come il blocco
    # Creiamo un dizionario colore per assicurarci che il testo abbia lo stesso colore del punto
    color_dict = dict(zip(unique_blocks, block_palette))

    # Usiamo adjust_text se disponibile, altrimenti testo standard
    texts = []
    for feature, row in top_loadings.iterrows():
        # Rimuoviamo il suffisso "_BlockX" dal nome visualizzato per pulizia nel grafico
        clean_name = feature.split('_Block')[0]
        block_color = color_dict.get(row['Block'], 'black')

        texts.append(ax.text(
            row[pc_x_label],
            row[pc_y_label],
            clean_name,
            fontsize=9,
            color=block_color,
            fontweight='bold'
        ))

    # --- 5. STYLING STANDARD ---
    # Linee centrali
    ax.axhline(0, color='k', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.axvline(0, color='k', linestyle='-', linewidth=0.8, alpha=0.5)

    # Limiti simmetrici per un aspetto quadrato
    max_val = max(abs(loadings[pc_x_label].max()), abs(loadings[pc_y_label].max())) * 1.1
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)

    # Labels con varianza spiegata
    var_x = pca_results['explained_variance'][pc_x - 1] * 100
    var_y = pca_results['explained_variance'][pc_y - 1] * 100

    ax.set_xlabel(f"Super Loading {pc_x_label} ({var_x:.1f}%)", fontweight='bold')
    ax.set_ylabel(f"Super Loading {pc_y_label} ({var_y:.1f}%)", fontweight='bold')
    ax.set_title(f"SUM-PCA Block Contribution: {pc_x_label} vs {pc_y_label}", fontweight='bold')

    # Legenda migliorata
    ax.legend(title="Data Block", loc='upper right', frameon=True)

    plt.tight_layout()

    # Salvataggio
    save_path = os.path.join(output_dir, f"{file_name}_PC{pc_x}vsPC{pc_y}.{SAVE_FORMAT}")
    plt.savefig(save_path, format=SAVE_FORMAT, bbox_inches='tight')
    plt.close(fig)
    print(f"Generating SUM-PCA Block Loading Plot saved to {save_path}")


def plot_loading_profile(pca_results, output_dir, pc_index=1, file_name="loading_profile"):
    """
    Generates a Spectral-style Line Plot for the loadings of a SINGLE component.

    THEORY:
    - Visualizes the 'shape' or 'profile' of the component across all variables.
    - Particularly useful in Omics (LC-MS, NMR) to identify spectral regions or
      clusters of m/z values that drive the separation.
    - X-axis: Features (ordered by m/z or index).
    - Y-axis: Loading weight.
    """
    os.makedirs(output_dir, exist_ok=True)

    pc_label = f"PC{pc_index}"
    # Retrieve loadings for the specific PC
    loadings = pca_results['loadings'][pc_label]

    fig, ax = plt.subplots()

    # --- 1. Determine X-Axis (Numeric m/z or Sequential) ---
    try:
        # Try to convert feature names to floats (assuming they represent m/z or RT)
        x_values = loadings.index.astype(float)
        # Sort by X to ensure the line doesn't scribble back and forth
        sorted_indices = np.argsort(x_values)
        x_plot = x_values[sorted_indices]
        y_plot = loadings.values[sorted_indices]
        xlabel = "Feature (m/z or RT)"
    except ValueError:
        # If features are strings (e.g., "Met_1"), use sequential index
        x_plot = np.arange(len(loadings))
        y_plot = loadings.values
        xlabel = "Feature Index"

    # --- 2. Color Selection ---
    # Cycle through the palette based on PC index to give distinct colors
    # e.g., PC1 -> Color[0], PC2 -> Color[1], etc.
    line_color = DISCRETE_COLORS[(pc_index - 1) % len(DISCRETE_COLORS)]

    # --- 3. Plotting (Line Plot) ---
    ax.plot(x_plot, y_plot, color=line_color, linewidth=1.2, label=f'Loadings {pc_label}')

    # Add a horizontal line at 0 for reference
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

    # --- 4. Styling ---
    ax.set_title(f"Loadings Profile: {pc_label}", fontweight='bold')
    ax.set_ylabel(f"Loading Value")
    ax.set_xlabel(xlabel)

    # Add grid but keep it subtle
    ax.grid(True, linestyle=':', alpha=0.4)

    # Optional: Fill area under curve slightly for better visual impact
    # ax.fill_between(x_plot, 0, y_plot, color=line_color, alpha=0.1)

    # Legend
    ax.legend(loc='upper right')

    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{file_name}_{pc_label}.{SAVE_FORMAT}")
    plt.savefig(save_path, format=SAVE_FORMAT, bbox_inches='tight')
    plt.close(fig)
    print(f"Generating PCA Loading Profile ({pc_label}) saved to {save_path}")


def detect_pca_outliers(pca_results, conf_level=0.95):
    """
    Performs Multivariate Anomaly Detection using PCA (Hotelling's T2 + Q-Residuals).
    Includes safety checks to prevent ZeroDivisionError on small class subsets.
    """

    # 1. Estrazione Dati
    scores_all = pca_results['scores'].values
    loadings_all = pca_results['loadings'].values
    eigenvalues_all = pca_results['model'].explained_variance_

    try:
        X_scaled = pca_results['data_scaled']
    except KeyError:
        raise KeyError("Missing 'data_scaled' in pca_results. Update perform_pca to return it.")

    n_samples = scores_all.shape[0]
    total_components = scores_all.shape[1]

    # --- 2. SELEZIONE INTELLIGENTE DELLE COMPONENTI (Safety Fix) ---
    # Non possiamo usare tutte le componenti, altrimenti T2 rompe la matematica (Division by Zero).
    # Strategia: Prendiamo le componenti che spiegano il 90% della varianza,
    # MA ci assicuriamo di lasciare almeno 2 gradi di libertà.

    # Calcolo varianza cumulativa locale
    explained_var_ratio = eigenvalues_all / np.sum(eigenvalues_all)
    cum_var = np.cumsum(explained_var_ratio)

    # Trova quante componenti servono per il 90% di varianza
    target_var = 0.90
    n_components_var = np.argmax(cum_var >= target_var) + 1

    # Calcolo limite massimo di sicurezza (n_samples - 2 è prudente per evitare divisioni per zero)
    # Esempio: se ho 40 campioni, uso max 38 componenti.
    max_safe_components = max(1, n_samples - 2)

    # Scegliamo il minimo tra le due logiche
    n_comp_final = min(n_components_var, max_safe_components, total_components)

    # Assicuriamoci di averne almeno 1 o 2 (se possibile)
    if n_comp_final < 1: n_comp_final = 1

    print(f"   [DEBUG] Detection using {n_comp_final}/{total_components} PCs (Samples: {n_samples})")

    # --- Tagliamo i dati in base alle componenti scelte ---
    scores = scores_all[:, :n_comp_final]
    loadings = loadings_all[:, :n_comp_final]
    eigenvalues = eigenvalues_all[:n_comp_final]

    # --- PART A: HOTELLING'S T2 ---
    # Formula: Sum( Score^2 / Eigenvalue )
    t2_values = np.sum((scores ** 2) / eigenvalues, axis=1)

    # Limite F-Distribution
    # Se n_samples è troppo piccolo rispetto a n_comp, questo crashava. Ora è protetto.
    d1 = n_comp_final
    d2 = n_samples - n_comp_final

    if d2 <= 0:
        # Caso limite estremo (non dovrebbe accadere col fix sopra, ma per sicurezza)
        print("   [WARNING] Degrees of freedom issue. T2 limit unreliable.")
        t2_limit = np.inf
    else:
        F_crit = f.ppf(conf_level, d1, d2)
        t2_limit = (d1 * (n_samples - 1) / d2) * F_crit

    # --- PART B: Q RESIDUALS ---
    # Errore di ricostruzione usando SOLO le componenti selezionate
    X_reconstructed = np.dot(scores, loadings.T)
    E_matrix = X_scaled - X_reconstructed
    q_values = np.sum(E_matrix ** 2, axis=1)

    # Q Limit (Weighted Chi-Square)
    mean_q = np.mean(q_values)
    var_q = np.var(q_values)

    if var_q > 0:
        g = var_q / (2 * mean_q)
        h = (2 * mean_q ** 2) / var_q
        q_limit = g * chi2.ppf(conf_level, df=h)
    else:
        q_limit = 0.0

    # --- PART C: RESULTS ---
    results_df = pd.DataFrame(index=pca_results['scores'].index)
    results_df['T2'] = t2_values
    results_df['T2_Limit'] = t2_limit
    results_df['Q_Residuals'] = q_values
    results_df['Q_Limit'] = q_limit

    results_df['Outlier_T2'] = results_df['T2'] > t2_limit
    results_df['Outlier_Q'] = results_df['Q_Residuals'] > q_limit
    results_df['is_outlier'] = results_df['Outlier_T2'] | results_df['Outlier_Q']

    def classify(row):
        if row['Outlier_T2'] and row['Outlier_Q']: return "Both"
        if row['Outlier_T2']: return "Hotelling (Extreme)"
        if row['Outlier_Q']: return "Q-Res (Model Mismatch)"
        return "Normal"

    results_df['Outlier_Type'] = results_df.apply(classify, axis=1)

    return results_df


def plot_distance_plot(pca_results, df, output_dir, highlight_samples=None, file_name="Distance_Plot"):
    """
    Generates an Distance Plot (Hotelling's T2 vs Q-Residuals).
    This is the GOLD STANDARD for deciding which outliers to remove.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Recupero Dati (Ricalcoliamo T2 e Q velocemente per coerenza)
    scores = pca_results['scores'].values
    loadings = pca_results['loadings'].values
    eigenvalues = pca_results['model'].explained_variance_
    X_scaled = pca_results['data_scaled']

    # Usiamo le componenti che spiegano il 90-95% (come nel detection)
    explained_var_ratio = eigenvalues / np.sum(eigenvalues)
    cum_var = np.cumsum(explained_var_ratio)
    n_comp = np.argmax(cum_var >= 0.90) + 1  # Usiamo 90% per coerenza col detection

    # Subset
    scores_sub = scores[:, :n_comp]
    loadings_sub = loadings[:, :n_comp]
    eigenvals_sub = eigenvalues[:n_comp]

    # Calcolo Metriche
    T2 = np.sum((scores_sub ** 2) / eigenvals_sub, axis=1)

    X_reconstructed = np.dot(scores_sub, loadings_sub.T)
    Q = np.sum((X_scaled - X_reconstructed) ** 2, axis=1)

    # Calcolo Limiti (Soglie)
    n = df.shape[0]
    conf = 0.95

    # Limite T2
    F_crit = f.ppf(conf, n_comp, n - n_comp)
    T2_lim = (n_comp * (n - 1) / (n - n_comp)) * F_crit

    # Limite Q
    mean_q = np.mean(Q)
    var_q = np.var(Q)
    g = var_q / (2 * mean_q)
    h = (2 * mean_q ** 2) / var_q
    Q_lim = g * chi2.ppf(conf, df=h)

    # --- PLOTTING ---
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter di tutti i punti (grigio chiaro)
    ax.scatter(T2, Q, c='grey', alpha=0.5, label='Samples')

    # Evidenziazione Outlier (quelli della tua lista)
    if highlight_samples:
        indices_to_plot = [i for i, name in enumerate(df.index) if name in highlight_samples]
        if indices_to_plot:
            ax.scatter(T2[indices_to_plot], Q[indices_to_plot], c='red', s=100, label='Out Candidates')

            # Etichette
            for idx in indices_to_plot:
                ax.text(T2[idx], Q[idx], df.index[idx], fontsize=9, color='red', fontweight='bold')

    # Linee Limite
    ax.axvline(T2_lim, color='blue', linestyle='--', label=f'T2 Limit (95%)')
    ax.axhline(Q_lim, color='green', linestyle='--', label=f'Q Limit (95%)')

    ax.set_xlabel("Hotelling's T2", fontweight='bold')
    ax.set_ylabel("Q-Residuals", fontweight='bold')
    ax.set_title(f"Distance Plot: {file_name}", fontweight='bold')
    ax.legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{file_name}.pdf")
    plt.savefig(save_path)
    plt.close()
    print(f"Distance Plot saved: {save_path}")


def biplot(df, output_dir, file_name="Biplot", class_col='Class', scaling='autoscaling', top_n=20):

    groups = df[class_col].values
    df_numeric = df.drop(columns=[class_col])
    contributors = top_n

    if scaling == 'autoscaling':
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_numeric)
    else:
        X_scaled = df_numeric.values

    pca = PCA(n_components=2)
    scores = pca.fit_transform(X_scaled)
    loadings = pca.components_.T
    exp_var = pca.explained_variance_ratio_ * 100

    # 4. Prepare DataFrame for Seaborn
    # This enables the use of hue/style mapping
    scores_df = pd.DataFrame(scores, columns=['PC1', 'PC2'])
    scores_df['Group'] = groups

    # 5. Setup Dynamic Palette & Markers (Global Config)
    # Get unique classes present in this specific dataset
    unique_classes = scores_df['Group'].unique()

    # Slice the global palette to match the number of classes
    current_palette = DISCRETE_COLORS[:len(unique_classes)]

    # Map markers dynamically using the global MARKERS list
    markers_dict = {cls: MARKERS[i % len(MARKERS)] for i, cls in enumerate(unique_classes)}

    # 6. Plotting
    fig, ax = plt.subplots(figsize=(10, 8))

    # --- SCORES (Samples) ---
    sns.scatterplot(data=scores_df, x='PC1', y='PC2',
                    hue='Group',
                    style='Group',
                    palette=current_palette,
                    markers=markers_dict,
                    s=100,
                    alpha=0.9,
                    edgecolor='k',
                    ax=ax)

    # --- LOADINGS (Vectors) ---
    # Scale vectors to match the visual range of the scores
    scale_factor = np.max(np.abs(scores)) * 0.8

    # Calculate magnitude to find top contributors
    magnitude = np.sqrt(loadings[:, 0] ** 2 + loadings[:, 1] ** 2)
    top_indices = np.argsort(magnitude)[-contributors:]

    metabolite_names = df_numeric.columns

    # Use gradient colors for vectors
    cmap = plt.get_cmap(SELECTED_PALETTE)
    colors_contributors = [cmap(i) for i in np.linspace(0.1, 0.9, len(top_indices))]

    for idx, i in enumerate(top_indices):
        x_end = loadings[i, 0] * scale_factor
        y_end = loadings[i, 1] * scale_factor
        c = colors_contributors[idx]
        feat_name = metabolite_names[i]

        # Draw Vector Line (Fixed: removed 'edgecolors' arg which caused the crash)
        ax.plot([0, x_end], [0, y_end], color=c, linewidth=1.5,
                path_effects=[pe.withStroke(linewidth=0.5, foreground='black')],
                alpha=0.8)

        # Draw Endpoint Dot
        ax.scatter(x_end, y_end, color=c, marker=MARKERS[3], edgecolors='black', linewidth=0.5, s=60, zorder=10,
                   label=feat_name)

        # Draw Label with Outline
        ha_align = 'left' if x_end > 0 else 'right'
        va_align = 'bottom' if y_end > 0 else 'top'

    ax.axhline(0, color='#050402', linestyle='solid', linewidth=0.8, alpha=0.8)
    ax.axvline(0, color='#050402', linestyle='solid', linewidth=0.8, alpha=0.8)
    ax.grid(color='gray', linestyle='dashed', linewidth=0.5, alpha=0.7)

    ax.set_xlabel(f'PC1 ({exp_var[0]:.2f}%)')
    ax.set_ylabel(f'PC2 ({exp_var[1]:.2f}%)')
    ax.set_title(f'Biplot: Samples & Top {contributors} contributors', fontweight='bold')

    # Place legend outside to prevent overlapping
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0, title=f"Classes and contributors")

    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{file_name}.{SAVE_FORMAT}")
    plt.savefig(save_path, format=SAVE_FORMAT, bbox_inches='tight')
    plt.close(fig)
    print(f"Generating PCA Loading Profile ({file_name}) saved to {save_path}")


