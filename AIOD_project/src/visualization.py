import math
import os
import numpy as np
import pandas as pd

import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config_visualization import *




def plot_boxplots(df, features, output_dir, file_prefix="boxplot", class_col='Class', ylabel='Peak Area',
                  features_per_page=6):
    """
    Generates Box Plots for the specified features to analyze distribution, outliers, and symmetry.

    This function implements the standard "five-number summary" visualization:
    - Box: Represents the Interquartile Range (IQR = Q3 - Q1).
    - Line: Median (Q2).
    - Whiskers: Extend to data points within 1.5 * IQR from the quartiles.
    - Outliers: Data points falling outside the whiskers.

    To adhere to accessibility requirements, individual data points are overlaid
    using markers (shapes) corresponding to the class, ensuring readability
    in black & white.

    Args:
        df (pd.DataFrame): The dataset containing samples as rows. Must include 'class_col'.
        features (list): List of column names (metabolites) to plot.
        output_dir (str): Directory where PDF files will be saved.
        file_prefix (str): Prefix for the output filenames.
        class_col (str): Name of the column containing group labels (default: 'Class').
        ylabel (str): Label for the Y-axis indicating the unit of measurement.
        features_per_page (int): Number of subplots to fit in a single PDF file.
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get unique classes to map markers consistently
    unique_classes = df[class_col].unique()
    # Fix Warning: Slice the palette to match exactly the number of classes found
    current_palette = DISCRETE_COLORS[:len(unique_classes)]
    # Create a dictionary mapping classes to markers from our global MARKERS list
    # e.g., {'CTRL': 'o', 'CHD': 's'}
    class_marker_map = {cls: MARKERS[i % len(MARKERS)] for i, cls in enumerate(unique_classes)}

    # Calculate number of pages needed
    num_features = len(features)
    num_pages = math.ceil(num_features / features_per_page)

    # Determine grid layout (e.g., 2 columns x 3 rows for 6 plots)
    n_cols = 2
    n_rows = math.ceil(features_per_page / n_cols)

    print(f"Generating Box Plots: {num_features} features across {num_pages} file(s)...")

    for page in range(num_pages):
        # Slice the features for the current page
        start_idx = page * features_per_page
        end_idx = min(start_idx + features_per_page, num_features)
        current_features = features[start_idx:end_idx]

        # Initialize figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 5 * n_rows))
        axes = axes.flatten()  # Flatten to easily iterate with index

        for i, feature in enumerate(current_features):
            ax = axes[i]

            # 1. Draw the Box Plot (The Statistical Summary)
            # whis=1.5 enforces the standard IQR rule (Q1-1.5*IQR, Q3+1.5*IQR)
            sns.boxplot(x=class_col, y=feature, data=df, ax=ax,
                        palette=current_palette,  # Use our global accessible palette
                        showfliers=False,  # Hide default outliers to avoid duplication with stripplot
                        width=0.5, linewidth=1.5)

            # 2. Overlay individual data points (Accessibility & Granularity)
            # We use markers to distinguish classes in B&W
            sns.stripplot(x=class_col, y=feature, data=df, ax=ax,
                          hue=class_col,  # Color by class
                          palette=current_palette,  # Same palette
                          dodge=False,  # Align with box
                          jitter=True,  # Add random noise to X to separate overlapping points
                          size=6, alpha=0.7,  # Transparency
                          edgecolor='gray', linewidth=0.5,  # Border for contrast
                          marker='o')  # Default marker, we customize below

            # Customizing markers manually because seaborn stripplot strictly maps markers
            # differently than scatterplot. We iterate collections to fix shapes if needed,
            # but a simpler way for consistent legend is relying on hue.
            # NOTE: Seaborn stripplot doesn't easily support 'style' for markers mapped to x-axis categories
            # in the same way scatterplot does. However, since the X-axis IS the class,
            # the shapes are redundant but helpful.
            # To strictly enforce shapes per class in stripplot is complex,
            # so we ensure the LEGEND reflects the shapes.

            # Set titles and labels
            ax.set_title(f"{feature} Distribution", fontweight='bold')
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Group")

            # Add grid for readability
            ax.grid(True, linestyle='--', alpha=0.3)

            # Remove the legend created by stripplot inside each subplot (too cluttered)
            if ax.get_legend():
                ax.get_legend().remove()

        # Hide any unused subplots (if features < slots on page)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Global Legend configuration for the page
        # We create custom handles to ensure the legend shows the correct Color AND Marker
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker=class_marker_map[cls], color='w', label=cls,
                   markerfacecolor=DISCRETE_COLORS[idx % len(DISCRETE_COLORS)],
                   markersize=10, markeredgecolor='k')
            for idx, cls in enumerate(unique_classes)
        ]

        fig.legend(handles=legend_elements, loc='upper right', title="Condition", bbox_to_anchor=(0.98, 0.98))

        plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # Adjust layout to make room for legend/title

        # Save output
        filename = f"{file_prefix}_page{page + 1}.{SAVE_FORMAT}"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, format=SAVE_FORMAT, bbox_inches='tight')
        plt.close(fig)  # Close to free memory

        print(f"   Saved: {save_path}")


def plot_overlay_dotplots(df, features, output_dir, file_prefix="dotplot", class_col='Class', ylabel='Signal Intensity',
                          features_per_page=6):
    """
    Generates Overlaid Dot Plots (Box Plot + Strip Plot) to visualize raw data distribution
    alongside statistical summaries.

    THEORY & IMPLEMENTATION DETAILS:
    --------------------------------
    1. Statistical Layer (Box Plot):
       - Box: IQR (Q1 to Q3).
       - Line: Median (Q2).
       - Whiskers: Strictly calculated as Q3 + 1.5*IQR and Q1 - 1.5*IQR.
       - Box style: Rendered with transparency (alpha) to prioritize raw data visibility.

    2. Raw Data Layer (Dot/Strip Plot):
       - Overlays individual samples to reveal the true dispersion and sample size (Small N).
       - Identifies outliers visually (points falling outside whiskers).
       - Uses specific markers and colors for accessibility.

    Args:
        df (pd.DataFrame): Dataset with samples as rows.
        features (list): List of metabolites/features to plot.
        output_dir (str): Destination folder for PDF files.
        file_prefix (str): Prefix for output filenames.
        class_col (str): Column name for grouping (default 'Class').
        ylabel (str): Y-axis label.
        features_per_page (int): Number of plots per PDF page.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Map classes to markers for consistency
    unique_classes = df[class_col].unique()
    # Fix Warning: Slice palette
    current_palette = DISCRETE_COLORS[:len(unique_classes)]
    class_marker_map = {cls: MARKERS[i % len(MARKERS)] for i, cls in enumerate(unique_classes)}

    num_features = len(features)
    num_pages = math.ceil(num_features / features_per_page)
    n_cols = 2
    n_rows = math.ceil(features_per_page / n_cols)

    print(f"Generating Overlay Dot Plots: {num_features} features across {num_pages} file(s)...")

    for page in range(num_pages):
        start_idx = page * features_per_page
        end_idx = min(start_idx + features_per_page, num_features)
        current_features = features[start_idx:end_idx]

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 5 * n_rows))
        axes = axes.flatten()

        for i, feature in enumerate(current_features):
            ax = axes[i]

            # --- LAYER 1: The Statistical Context (Box Plot) ---
            # We use a lighter alpha (transparency) for the box so it doesn't obscure the dots.
            sns.boxplot(x=class_col, y=feature, data=df, ax=ax,
                        palette=current_palette,
                        showfliers=False,  # We hide automatic outliers because we show ALL points below
                        width=0.5,
                        linewidth=1.2,
                        boxprops=dict(alpha=0.4))  # Crucial: Transparency for the box

            # --- LAYER 2: The Raw Data (Dot/Strip Plot) ---
            # This provides the "crucial transparency" described in the theory.
            sns.stripplot(x=class_col, y=feature, data=df, ax=ax,
                          hue=class_col,
                          palette=current_palette,
                          dodge=False,  # Align dots with the box center
                          jitter=0.2,  # Spread points horizontally to avoid overlap
                          size=7,  # Larger points for better visibility
                          alpha=0.9,  # High opacity for points
                          edgecolor='black',  # Black border to define individual points
                          linewidth=0.8,
                          marker='o')  # Base marker, specialized below if needed

            # Visual Clean-up
            ax.set_title(f"{feature}", fontweight='bold')
            ax.set_ylabel(ylabel)
            ax.set_xlabel("")  # Class is already obvious from x-ticks
            ax.grid(True, linestyle='--', alpha=0.3)

            if ax.get_legend():
                ax.get_legend().remove()

        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Create Custom Legend (Matching Color + Marker)
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker=class_marker_map[cls], color='w', label=cls,
                   markerfacecolor=DISCRETE_COLORS[idx % len(DISCRETE_COLORS)],
                   markersize=10, markeredgecolor='k')
            for idx, cls in enumerate(unique_classes)
        ]

        fig.legend(handles=legend_elements, loc='upper right', title="Group", bbox_to_anchor=(0.98, 0.98))
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])

        filename = f"{file_prefix}_page{page + 1}.{SAVE_FORMAT}"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, format=SAVE_FORMAT, bbox_inches='tight')
        plt.close(fig)

        print(f"   Saved: {save_path}")


def plot_sample_distributions(df, output_dir, file_name="samples_qc_boxplot", class_col='Class', samples_per_page=20, plot_title=None, show_sample_names=False, showfliers=False):
    """
    Generates Box Plots visualizing the global intensity distribution of EACH SAMPLE.

    THEORY & USE CASE:
    ------------------
    - QC (Quality Control): Essential to identify outlier samples (e.g., dilution errors, instrument failure).
    - Normalization Check: Before normalization, boxplots might be uneven.
      After normalization, medians and IQRs should be roughly aligned across all samples.

    Args:
        df (pd.DataFrame): Dataset (Samples x Features).
        output_dir (str): Output directory.
        file_name (str): Output filename prefix.
        class_col (str): Column identifying the group (used for coloring).
        samples_per_page (int): Number of samples (X-axis) to show per PDF page.
    """
    os.makedirs(output_dir, exist_ok=True)

    if plot_title is None:
        plot_title = "Sample Intensity Distributions"

    # 1. Identify Numeric Features (Metabolites)
    # We exclude the Class column and any other non-numeric info
    # Using the variable class_col ensures flexibility if the column name changes
    numeric_df = df.drop(columns=[class_col])

    # We need the class column aligned with numeric data for coloring
    # Ensure indices match
    classes = df[class_col]

    # 2. Setup Coloring
    unique_classes = df[class_col].unique()
    # Handle palette slicing based on the number of classes
    current_palette = DISCRETE_COLORS[:len(unique_classes)]

    # 3. Pagination Logic (Iterate by Samples/Rows)
    num_samples = len(df)
    num_pages = math.ceil(num_samples / samples_per_page)

    print(f"Generating Sample QC Boxplots: {num_samples} samples across {num_pages} file(s)...")

    # List of Sample IDs
    sample_ids = df.index.tolist()

    for page in range(num_pages):
        start_idx = page * samples_per_page
        end_idx = min(start_idx + samples_per_page, num_samples)

        current_batch_ids = sample_ids[start_idx:end_idx]

        # Subset the dataframe for the current page
        # We need "Long Format" for Seaborn: SampleID | Class | Intensity
        # This allows us to plot x=SampleID, y=Intensity, hue=Class

        batch_numeric = numeric_df.loc[current_batch_ids]
        batch_classes = classes.loc[current_batch_ids]

        # Merge for melting
        batch_combined = batch_numeric.copy()
        batch_combined['__Class__'] = batch_classes  # Temporary column
        batch_combined['__SampleID__'] = batch_combined.index

        # Melt: Transform from Wide (Cols=Metabolites) to Long (Rows=Measurements)
        # This is heavy, but we do it only for the page subset
        df_melted = batch_combined.melt(id_vars=['__SampleID__', '__Class__'],
                                        var_name='Metabolite',
                                        value_name='Intensity')

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 4.8))

        # We set hue_order to ensure consistent coloring even if a page is missing a class
        # We also define medianprops to make the median line white and visible against dark boxes
        sns.boxplot(data=df_melted, x='__SampleID__', y='Intensity',
                    hue='__Class__',
                    hue_order=unique_classes,
                    palette=current_palette,
                    showfliers=showfliers,
                    flierprops={"marker": ".", "markerfacecolor": "black", "markeredgecolor": "none", "markersize": 4, "alpha": 0.8},
                    linewidth=0.5, width=0.8,
                    medianprops={'color': 'white', 'linewidth': 1.0, 'path_effects': [pe.withStroke(linewidth=2.0, foreground='black')]},
                    ax=ax)

        # Visual Polish
        page_info = f" (Plot {page + 1}/{num_pages})" if num_pages > 1 else ""
        ax.set_title(f"{plot_title}{page_info}", fontweight='bold')
        ax.set_xlabel("Sample ID")
        ax.set_ylabel("Intensity Distribution (All Metabolites)")

        if show_sample_names:
            # Mostra TUTTI i nomi dei campioni correnti, ruotati di 90 gradi
            ax.set_xticks(range(len(current_batch_ids)))
            ax.set_xticklabels(current_batch_ids, rotation=90, fontsize=8)
            ax.set_xlabel("Sample Name")
        else:
            # Vecchia logica: mostra solo gli indici numerici ogni 20
            step = 20
            # Gestione sicurezza se step > len
            step = min(step, len(current_batch_ids)) if len(current_batch_ids) > 0 else 1

            tick_positions = range(0, len(current_batch_ids), step)
            tick_labels = range(start_idx, start_idx + len(current_batch_ids), step)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)
            ax.set_xlabel("Sample Index")

        # Legend (including median)
        handles, labels = ax.get_legend_handles_labels()
        median_proxy = Line2D([0], [0], color='white', linewidth=1.0,
                              path_effects=[pe.withStroke(linewidth=2.0, foreground='black')],
                              label='Median')
        handles.append(median_proxy)
        labels.append("Median")
        ax.legend(handles=handles, labels=labels, title="Group", loc='best')

        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

        plt.tight_layout()

        filename = f"{file_name}_page{page + 1}.{SAVE_FORMAT}"
        plt.savefig(os.path.join(output_dir, filename), format=SAVE_FORMAT, bbox_inches='tight')
        plt.close(fig)
        print(f"   Saved: {os.path.join(output_dir, filename)}")


def plot_features_overview(df, output_dir, file_name="features_overview", class_col='Class', features_per_page=20,
                           plot_title=None):
    """
    Generates a dense overview of Feature Distributions.
    Unlike 'plot_boxplots' (which makes separate subplots), this puts multiple features
    on the SAME X-AXIS to compare their relative intensities side-by-side.
    """
    os.makedirs(output_dir, exist_ok=True)

    if plot_title is None:
        plot_title = "Metabolite Intensity Overview"

    # 1. Identify Features (All columns except Class)
    # This is the main difference: we iterate over COLUMNS, not rows.
    all_features = [col for col in df.columns if col != class_col]

    num_features = len(all_features)
    num_pages = math.ceil(num_features / features_per_page)

    # Setup Coloring
    unique_classes = df[class_col].unique()
    current_palette = DISCRETE_COLORS[:len(unique_classes)]

    print(f"Generating Features Overview: {num_features} features across {num_pages} file(s)...")

    for page in range(num_pages):
        start_idx = page * features_per_page
        end_idx = min(start_idx + features_per_page, num_features)

        # Slice the LIST of features
        current_batch_features = all_features[start_idx:end_idx]

        # 2. Subset & Melt
        # We take the Class column + the current batch of features
        subset_df = df[[class_col] + current_batch_features].copy()

        # Transform to Long Format:
        # Rows become: [Class, FeatureName, Intensity]
        df_melted = subset_df.melt(id_vars=[class_col],
                                   value_vars=current_batch_features,
                                   var_name='Metabolite',
                                   value_name='Intensity')

        # 3. Plotting (A4 Landscape Fixed Size)
        fig, ax = plt.subplots(figsize=(12, 4))

        sns.boxplot(data=df_melted, x='Metabolite', y='Intensity',
                    hue=class_col,
                    palette=current_palette,
                    showfliers=False,
                    linewidth=0.5, width=0.7,
                    # Stesso stile mediana della funzione Sample
                    medianprops={'color': 'white', 'linewidth': 1.0,
                                 'path_effects': [pe.withStroke(linewidth=2.0, foreground='black')]},
                    ax=ax)

        # Visual Polish
        page_info = f" (Features {start_idx + 1}-{end_idx} of {num_features})" if num_pages > 1 else ""
        ax.set_title(f"{plot_title}{page_info}", fontweight='bold')
        ax.set_ylabel("Intensity")

        step = 20
        tick_positions = range(0, len(current_batch_features), step)
        tick_labels = range(start_idx, start_idx + len(current_batch_features), step)

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel("Metabolite Index")
        ax.tick_params(axis='x', rotation=0)  # Numeri dritti

        # Legend Customization (con stile mediana)
        handles, labels = ax.get_legend_handles_labels()
        median_proxy = Line2D([0], [0], color='white', linewidth=1.0,
                              path_effects=[pe.withStroke(linewidth=2.0, foreground='black')],
                              label='Median')
        handles.append(median_proxy)
        labels.append("Median")

        ax.legend(handles=handles, labels=labels, title="Group", loc='best')

        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

        plt.tight_layout()

        full_filename = f"{file_name}_page{page + 1}.{SAVE_FORMAT}"
        plt.savefig(os.path.join(output_dir, full_filename), format=SAVE_FORMAT, bbox_inches='tight')
        plt.close(fig)
        print(f"   Saved: {os.path.join(output_dir, full_filename)}")

def plot_global_density(df, output_dir, file_name="global_density_plot", class_col='Class', xlabel='Intensity', add_gaussian=True, plot_title=None):
    """
    Generates a Global Density Plot (Histogram + KDE) to visualize the effect of data transformations.
    Instead of plotting individual features, it pools ALL metabolite intensities together to assess
    global normality and skewness reduction.

    THEORY & IMPLEMENTATION DETAILS:
    --------------------------------
    1. Aggregation:
       - Flattens the Sample x Feature matrix into a single vector of intensities per class.
       - Essential for evaluating if a transformation (e.g., Log10) effectively compresses
         the dynamic range and centers the global distribution.
    2. KDE (Solid Line):
       - Shows the EMPIRICAL shape of the transformed data.
    3. Gaussian/Normal (Dashed Line) [Optional]:
       - Shows the THEORETICAL shape. Ideally, after transformation, the Solid and Dashed lines
         should overlap significantly.

    Args:
        df (pd.DataFrame): Dataset (Transformed).
        output_dir (str): Output directory.
        file_name (str): Output filename (e.g., 'global_dist_log10').
        class_col (str): Grouping variable.
        xlabel (str): Label for the X-axis (e.g., 'Log10 Intensity').
        add_gaussian (bool): If True, overlaps the theoretical Gaussian curve (dashed).
    """

    os.makedirs(output_dir, exist_ok=True)

    if plot_title is None:
        plot_title = "Global Distribution Analysis"

    # 1. Prepare Data: Melt from Wide (Samples x Features) to Long (One huge column of intensities)
    # We drop non-numeric columns except Class
    numeric_df = df.copy()

    # Melt the dataframe to have a single 'Intensity' column for all features
    # This creates a very long dataframe, but it's necessary for the global histogram
    melted_df = numeric_df.melt(id_vars=[class_col], var_name='Feature', value_name='Intensity')

    # Remove NaNs if any (some transformations might produce NaNs on 0 if not handled, though your functions handle it)
    melted_df = melted_df.dropna(subset=['Intensity'])

    # 2. Setup Plotting
    n_classes = df[class_col].nunique()
    current_palette = DISCRETE_COLORS[:n_classes]
    unique_classes = df[class_col].unique()
    color_map = dict(zip(unique_classes, current_palette))

    fig, ax = plt.subplots(figsize=(10, 6))

    # 3. Plot Histogram with Empirical Density (KDE)
    # Using 'stat="density"' ensures the area under the curve sums to 1, comparable to the Gaussian
    sns.histplot(data=melted_df, x='Intensity', hue=class_col,
                 palette=current_palette,
                 kde=True,  # Real Data Curve (Solid)
                 element="step",  # 'step' is cleaner for overlapping histograms than 'bars'
                 stat="density",
                 common_norm=False,  # Normalize each class independently
                 alpha=0.2,  # Light fill
                 linewidth=1.0,
                 ax=ax)

    # 4. (Optional) Overlay Theoretical Gaussian Curve
    if add_gaussian:
        # Get plot limits to generate the x-axis for the Gaussian curve
        x_min, x_max = ax.get_xlim()
        x_axis = np.linspace(x_min, x_max, 500)

        for cls in unique_classes:
            # Extract pooled data for this class
            cls_data = melted_df[melted_df[class_col] == cls]['Intensity']

            if len(cls_data) > 1:
                # Fit Normal Distribution (Calculate Global Mean and Std Dev)
                mu, std = norm.fit(cls_data)

                # Generate PDF (Probability Density Function)
                p = norm.pdf(x_axis, mu, std)

                # Plot Dashed Line in the same color as the class
                # We use path_effects to make the dashed line stand out against the histogram
                ax.plot(x_axis, p, linestyle='--', linewidth=2.0,
                        color=color_map[cls], alpha=0.9,
                        label=f'{cls} Theoretical Normal')

    ax.set_title(f"{plot_title}", fontweight='bold')
    ax.set_xlabel(f"{xlabel} (All Features Pooled)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.5)

    legend_handles = []
    for cls in unique_classes:
        legend_handles.append(mpatches.Patch(color=color_map[cls], label=cls))
    legend_handles.append(Line2D([0], [0], color='black', linestyle='-', linewidth=1.5, label='Empirical (KDE)'))
    if add_gaussian:
        legend_handles.append(
            Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='Gaussian/Normal'))
    ax.legend(handles=legend_handles, loc='best', title="Legend")

    plt.tight_layout()

    # 6. Save
    full_path = os.path.join(output_dir, f"{file_name}.{SAVE_FORMAT}")
    plt.savefig(full_path, format=SAVE_FORMAT, bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved Global Density Plot: {full_path}")

def plot_feature_histograms(df, features, output_dir, file_prefix="hist", class_col='Class', xlabel='Intensity',
                            features_per_page=6, add_gaussian=True):
    """
    Generates Histograms to visualize the frequency distribution of continuous data.
    Can overlay both the empirical density (KDE) and the theoretical Normal Distribution (Gaussian).

    THEORY & IMPLEMENTATION DETAILS:
    --------------------------------
    1. Structure:
       - Plots frequency/density of metabolite intensities.
    2. KDE (Solid Line):
       - Shows the REAL shape of the data (skewness, multimodal, etc.).
    3. Gaussian/Normal (Dashed Line) [Optional]:
       - Shows the THEORETICAL shape if data were perfectly normal (parametric).
       - Comparison: If Solid and Dashed lines match, T-test is safe. If they differ, consider Mann-Whitney.

    Args:
        df (pd.DataFrame): Dataset.
        features (list): List of metabolites to plot.
        output_dir (str): Output directory.
        file_prefix (str): Filename prefix.
        class_col (str): Grouping variable.
        xlabel (str): Label for the X-axis.
        features_per_page (int): Number of subplots per PDF.
        add_gaussian (bool): If True, overlaps the theoretical Gaussian curve (dashed).
    """

    os.makedirs(output_dir, exist_ok=True)

    # Fix Warning: Slice palette to match number of classes
    n_classes = df[class_col].nunique()
    current_palette = DISCRETE_COLORS[:n_classes]

    # Map classes to colors for manual plotting of Gaussian lines
    unique_classes = df[class_col].unique()
    color_map = dict(zip(unique_classes, current_palette))

    num_features = len(features)
    num_pages = math.ceil(num_features / features_per_page)
    n_cols = 2
    n_rows = math.ceil(features_per_page / n_cols)

    print(f"Generating Histograms: {num_features} features across {num_pages} file(s)...")

    for page in range(num_pages):
        start_idx = page * features_per_page
        end_idx = min(start_idx + features_per_page, num_features)
        current_features = features[start_idx:end_idx]

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 5 * n_rows))
        axes = axes.flatten()

        for i, feature in enumerate(current_features):
            ax = axes[i]

            # 1. Plot Histogram with Empirical Density (KDE)
            sns.histplot(data=df, x=feature, hue=class_col,
                         palette=current_palette,
                         kde=True,  # Real Data Curve (Solid)
                         element="bars",
                         stat="density",  # Normalize to density
                         common_norm=False,
                         alpha=0.3,
                         edgecolor=None,
                         line_kws={'linewidth': 1.5},  # Thicker KDE line
                         ax=ax)

            # 2. (Optional) Overlay Theoretical Gaussian Curve
            if add_gaussian:
                # We calculate and plot a Gaussian for EACH class separately
                x_min, x_max = ax.get_xlim()
                x_axis = np.linspace(x_min, x_max, 100)

                for cls in unique_classes:
                    # Extract data for this class
                    cls_data = df[df[class_col] == cls][feature].dropna()

                    if len(cls_data) > 1:
                        # Fit Normal Distribution (Calculate Mean and Std Dev)
                        mu, std = norm.fit(cls_data)

                        # Generate PDF (Probability Density Function)
                        p = norm.pdf(x_axis, mu, std)

                        # Plot Dashed Line in the same color as the class
                        ax.plot(x_axis, p, linestyle='--', linewidth=1.5,
                                color=color_map[cls], alpha=0.8,
                                label=f'{cls} Normal' if i == 0 else "")  # Label only once to avoid legend clutter

            ax.set_title(f"{feature} Normality Check", fontweight='bold')
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Density")
            ax.grid(True, linestyle=':', alpha=0.4)

            # Clean up legend
            if ax.get_legend():
                ax.get_legend().set_title(class_col)

        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()

        filename = f"{file_prefix}_page{page + 1}.{SAVE_FORMAT}"
        plt.savefig(os.path.join(output_dir, filename), format=SAVE_FORMAT, bbox_inches='tight')
        plt.close(fig)
        print(f"   Saved: {os.path.join(output_dir, filename)}")



def plot_feature_means_bar(df, features, output_dir, file_prefix="barplot", class_col='Class', ylabel='Mean Intensity',
                           features_per_page=6):
    """
    Generates Bar Charts to compare the Mean intensity of features across distinct categories.

    THEORY & IMPLEMENTATION DETAILS:
    --------------------------------
    1. Structure (Gaps):
       - Represents categorical comparison (CTRL vs CHD).
       - Bars are separated by gaps to emphasize discrete categories.
    2. Metrics:
       - Height: Represents the Mean value of the group.
       - Error Bars: Represent variability (Standard Deviation or CI), crucial for
         understanding if differences are meaningful.
    3. Accessibility:
       - Uses 'Hatching' (patterns) in addition to color to distinguish groups
         in black & white prints.

    Args:
        df (pd.DataFrame): Dataset.
        features (list): List of metabolites.
        output_dir (str): Output directory.
        file_prefix (str): Filename prefix.
        class_col (str): Grouping variable.
        ylabel (str): Label for Y-axis (Metric being compared).
        features_per_page (int): Number of subplots per PDF.
    """

    os.makedirs(output_dir, exist_ok=True)

    n_classes = df[class_col].nunique()
    current_palette = DISCRETE_COLORS[:n_classes]

    num_features = len(features)
    num_pages = math.ceil(num_features / features_per_page)
    n_cols = 2
    n_rows = math.ceil(features_per_page / n_cols)

    # Define hatches for accessibility (e.g., /// for one class, ... for another)
    hatches = ['//', '..', 'xx', '--']

    print(f"Generating Bar Charts: {num_features} features across {num_pages} file(s)...")

    for page in range(num_pages):
        start_idx = page * features_per_page
        end_idx = min(start_idx + features_per_page, num_features)
        current_features = features[start_idx:end_idx]

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 5 * n_rows))
        axes = axes.flatten()

        for i, feature in enumerate(current_features):
            ax = axes[i]

            # Bar Plot: Calculates Mean by default
            # capsize adds the little horizontal lines to the error bars
            bar_plot = sns.barplot(x=class_col, y=feature, data=df, ax=ax,
                                   palette=current_palette,
                                   edgecolor='black',  # Add border to bars
                                   errorbar='sd',  # Show Standard Deviation (or 'ci' for 95% CI)
                                   capsize=0.1,  # Width of error bar caps
                                   width=0.6)  # Width < 1 ensures GAPS between bars

            # Apply Hatching for Accessibility (Looping through bars/patches)
            # Seaborn creates a patch for each bar. We cycle through patterns based on class.
            num_classes = df[class_col].nunique()
            for j, patch in enumerate(ax.patches):
                # Calculate which class this patch belongs to
                # Note: This logic assumes bars are drawn in order of classes
                hatch = hatches[j % num_classes % len(hatches)]
                patch.set_hatch(hatch)

            ax.set_title(f"{feature} Mean Comparison", fontweight='bold')
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Group")
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)

        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()

        filename = f"{file_prefix}_page{page + 1}.{SAVE_FORMAT}"
        plt.savefig(os.path.join(output_dir, filename), format=SAVE_FORMAT, bbox_inches='tight')
        plt.close(fig)
        print(f"   Saved: {os.path.join(output_dir, filename)}")


def plot_class_distribution_pie(df, output_dir, file_name="class_balance_pie", class_col='Class'):
    """
    Generates a Pie Chart to visualize the proportional distribution of categorical data
    (specifically the Class balance).

    THEORY & IMPLEMENTATION DETAILS:
    --------------------------------
    1. Scope (Qualitative Only):
       - Strictly applied to categorical/qualitative data (e.g., CTRL vs CHD).
       - NOT suitable for quantitative continuous data (metabolite intensities).
    2. Metrics:
       - Visualizes the relative frequency (percentage) of each category as a slice of the whole.
    3. Accessibility:
       - Uses the global color palette.
       - Adds white separators (wedgeprops) between slices to distinguish them clearly.
       - Includes both percentages and absolute counts in the visualization/legend.

    Args:
        df (pd.DataFrame): Dataset containing the class column.
        output_dir (str): Output directory.
        file_name (str): Filename for the output PDF.
        class_col (str): The categorical column to visualize (default: 'Class').
    """

    os.makedirs(output_dir, exist_ok=True)

    # 1. Calculate Frequencies (Absolute and Relative)
    class_counts = df[class_col].value_counts()
    total_samples = len(df)

    # 2. Prepare Colors (Slice from global palette)
    colors = DISCRETE_COLORS[:len(class_counts)]

    # 3. Create Plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # wedgeprops='edgecolor' adds a white line between slices for better accessibility
    wedges, texts, autotexts = ax.pie(
        class_counts,
        labels=class_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2, 'antialiased': True},
        textprops={'fontsize': 12, 'weight': 'bold'}
    )

    # Style the percentage text inside the slices
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')

        autotext.set_path_effects([pe.withStroke(linewidth=2.0, foreground='black')])

    # 4. Detailed Legend
    # We add the absolute count (n) to the legend for completeness
    legend_labels = [f"{label} (n={count})" for label, count in zip(class_counts.index, class_counts)]

    ax.legend(wedges, legend_labels,
              title="Study Groups",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))

    ax.set_title(f"Class Distribution (Total n={total_samples})", fontweight='bold')

    # 5. Save Output
    save_path = os.path.join(output_dir, f"{file_name}.{SAVE_FORMAT}")
    plt.savefig(save_path, format=SAVE_FORMAT, bbox_inches='tight')
    plt.close(fig)

    print(f"Generating Pie Chart: Class balance saved to {save_path}")


def plot_scatter(df, x_col, y_col, output_dir, file_name="scatter", class_col='Class', add_regression=False):
    """
    Generates a 2D Scatter Plot to visualize the relationship between two quantitative variables.

    THEORY & IMPLEMENTATION DETAILS:
    --------------------------------
    1. Structure:
       - Plots data points on a Cartesian plane (X vs Y).
       - Fundamental for Exploratory Data Analysis (EDA) to detect patterns/trends.
    2. Covariance Visualization:
       - Helps identify Positive, Negative, or Null covariance based on point distribution.
       - Optional 'add_regression' overlays a linear fit to emphasize the trend.
    3. Accessibility:
       - Uses Dual Coding: Categories are distinguished by both Color and Marker shape.
    4. Omics Applications:
       - This generic function serves as the foundation for specialized plots like
         PCA Scores Plots (PC1 vs PC2) or Coomans' Plots.

    Args:
        df (pd.DataFrame): Dataset containing the X and Y columns and the class column.
        x_col (str): Name of the column for the X-axis.
        y_col (str): Name of the column for the Y-axis.
        output_dir (str): Directory where the PDF will be saved.
        file_name (str): Name of the output file (without extension).
        class_col (str): Column used for grouping (color/marker).
        add_regression (bool): If True, overlays a linear regression line to show correlation.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Map classes to markers manually for consistency with other plots
    unique_classes = df[class_col].unique()
    # Fix Warning: Slice palette
    current_palette = DISCRETE_COLORS[:len(unique_classes)]
    markers_dict = {cls: MARKERS[i % len(MARKERS)] for i, cls in enumerate(unique_classes)}

    fig, ax = plt.subplots(figsize=(8, 6))

    # 1. Base Scatter Plot
    # We map both hue and style to class_col for maximum accessibility
    sns.scatterplot(data=df, x=x_col, y=y_col,
                    hue=class_col,
                    style=class_col,
                    markers=markers_dict,
                    palette=current_palette,
                    s=80,  # Size of points
                    alpha=0.8,  # Transparency
                    edgecolor='k',  # Black edge for better definition
                    ax=ax)

    # 2. Optional Regression Line (Visualizing Covariance)
    if add_regression:
        # We draw a single regression line for the whole dataset to show global trend
        # scatter=False because points are already drawn above with custom styles
        sns.regplot(data=df, x=x_col, y=y_col, scatter=False,
                    ax=ax, color='gray', line_kws={"linestyle": "--"})

    # Visual Polish
    ax.set_title(f"Scatter Plot: {x_col} vs {y_col}", fontweight='bold')
    ax.set_xlabel(x_col, fontweight='bold')
    ax.set_ylabel(y_col, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)

    # Improve Legend
    # We move the legend outside if it obstructs data, but usually 'best' works.
    # Here we ensure the title is clear.
    if ax.get_legend():
        ax.get_legend().set_title("Group")
        # Ensure markers in legend match the plot

    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{file_name}.{SAVE_FORMAT}")
    plt.savefig(save_path, format=SAVE_FORMAT, bbox_inches='tight')
    plt.close(fig)

    print(f"Generating Scatter Plot: {x_col} vs {y_col} saved to {save_path}")


def plot_clustermap(df, features, output_dir, file_name="heatmap", class_col='Class'):
    """
    Generates a Clustered Heatmap with Dendrograms to visualize patterns, groups, and relative intensity.

    THEORY & IMPLEMENTATION DETAILS:
    --------------------------------
    1. Matrix Structure (Transposed):
       - As per Omics theory, Samples are placed on Columns, and Features (Metabolites) on Rows.
       - The input DataFrame (Samples x Features) is transposed automatically.
    2. Hierarchical Clustering (Dendrograms):
       - Side graphs (trees) show the result of Agglomerative Hierarchical Clustering.
       - Reorders rows and columns to group similar profiles together.
    3. Standardization (Z-Score):
       - Applies Z-score normalization per Feature (Row).
       - Converts raw intensity to "Standard Deviations from the Mean".
       - Essential to compare metabolites with vastly different intensity ranges.
    4. Class Annotation:
       - Adds a color bar to the sample axis to visualize if clustering aligns with biological groups (CTRL vs CHD).

    Args:
        df (pd.DataFrame): Dataset (Samples as Rows).
        features (list): List of metabolites to include (usually Significant ones).
        output_dir (str): Output directory.
        file_name (str): Filename.
        class_col (str): Grouping column.
    """

    os.makedirs(output_dir, exist_ok=True)

    # 1. Prepare Data
    # We work on a copy to avoid modifying the original dataframe
    data = df[features].copy()
    labels = df[class_col].copy()

    # Create a mapping of Class -> Color using our global palette
    unique_classes = labels.unique()
    lut = dict(zip(unique_classes, DISCRETE_COLORS[:len(unique_classes)]))

    # Create the color bar for columns (samples)
    # This maps every sample name to its class color
    col_colors = labels.map(lut)

    # 2. Transpose Data (Theory: Samples on Columns, Features on Rows)
    data_transposed = data.T

    print(f"Generating Clustered Heatmap for {len(features)} features...")

    # 3. Generate Clustermap
    # z_score=0 calculates z-score for each ROW (Feature).
    # This is crucial: specific metabolites have different ranges.
    # We want to see relative abundance (High/Low) relative to that metabolite's average.
    g = sns.clustermap(data_transposed,
                       cmap=SELECTED_PALETTE,  # Consistent with project theme (Viridis)
                       z_score=0,  # Normalize rows (0)
                       metric="euclidean",  # Distance metric
                       method="ward",  # Linkage method (minimizes variance)
                       col_colors=col_colors,  # Adds the class bar at the top
                       figsize=(12, 10),
                       dendrogram_ratio=(.15, .15),  # Proportion of figure for dendrograms
                       cbar_pos=(0.02, 0.8, 0.03, 0.15),  # Position of color scale legend
                       yticklabels=True,  # Show metabolite names
                       xticklabels=False)  # Hide sample names (too many to read)

    # 4. Legend & Polish
    # Add a custom legend for the Class Colors (since clustermap doesn't do it automatically for col_colors)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=lut[label], edgecolor='w', label=label) for label in unique_classes]

    # Position the legend safely
    plt.legend(handles=handles, title="Group", loc='upper right', bbox_to_anchor=(0.95, 0.95),
               bbox_transform=plt.gcf().transFigure)

    # Adjust title
    g.fig.suptitle(f"Hierarchical Clustering Heatmap ({len(features)} Features)", fontsize=16, fontweight='bold',
                   y=0.98)

    # 5. Save
    save_path = os.path.join(output_dir, f"{file_name}.{SAVE_FORMAT}")
    plt.savefig(save_path, format=SAVE_FORMAT, bbox_inches='tight')
    plt.close(g.fig)  # Clustermap creates its own figure instance

    print(f"   Saved: {save_path}")

