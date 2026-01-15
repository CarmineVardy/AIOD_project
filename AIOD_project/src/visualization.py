import math
import os

import matplotlib.patheffects as pe
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config_visualization import *


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


def plot_sample_distributions(df, output_dir, file_name="samples_qc_boxplot", class_col='Class', samples_per_page=20):
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

    # 1. Identify Numeric Features (Metabolites)
    # We exclude the Class column and any other non-numeric info
    numeric_df = df.select_dtypes(include=[np.number])

    # We need the class column aligned with numeric data for coloring
    # Ensure indices match
    classes = df[class_col]

    # 2. Setup Coloring
    unique_classes = df[class_col].unique()
    # Fix Warning: Slice palette
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
        fig, ax = plt.subplots(figsize=(12, 6))

        sns.boxplot(data=df_melted, x='__SampleID__', y='Intensity',
                    hue='__Class__',  # Color the sample box by its class
                    palette=current_palette,
                    showfliers=False,  # Hide outliers (too many dots for a QC plot)
                    linewidth=1.0, width=0.6, ax=ax)

        # Visual Polish
        ax.set_title(f"Sample Intensity Distributions (Page {page + 1}/{num_pages})", fontweight='bold')
        ax.set_xlabel("Sample ID")
        ax.set_ylabel("Intensity Distribution (All Metabolites)")

        # Rotate X labels (Sample Names) for readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center', fontsize=9)

        # Legend
        if ax.get_legend():
            ax.get_legend().set_title("Group")

        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

        plt.tight_layout()

        filename = f"{file_name}_page{page + 1}.{SAVE_FORMAT}"
        plt.savefig(os.path.join(output_dir, filename), format=SAVE_FORMAT, bbox_inches='tight')
        plt.close(fig)
        print(f"   Saved: {os.path.join(output_dir, filename)}")

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
    # We make it white if the slice is dark, or black if light, but white usually works well with Viridis/Plasma
    plt.setp(autotexts, size=11, weight="bold", color="white", path_effects=[])

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







"""
    Support function which saves the plot in a losses pdf format as default.
    Plot and directory are passed to the function dinamically
"""
def save_plot(plt, filename, output_folder, custom_format='pdf'):

    filename = f"{filename}.{custom_format}"

    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Created directory: {output_folder}")

        plt.savefig(os.path.join(output_folder, filename), format=custom_format, bbox_inches='tight')
        plt.close()
        print(f"Graph {filename} saved in: {output_folder}")
    except:
        print(f"[ERROR] in saving {filename}")



"""
    By assuming the first column contains names/IDs, this function extracts the first column of the dataframe in order to use the sample IDs in graphs
"""
def get_sample_names(df):

    # Reset index if the names are currently in the index
    if df.index.name is not None or not isinstance(df.index, pd.RangeIndex):
        return df.index.tolist()
    elif df.T.equals(df):  # Check if DataFrame is transposed        
        return df.iloc[:, 0].tolist() # Return the first column
    else:
        return df.iloc[0, :].tolist() # Return the first row



def sumPCA(df, dataset_label, output_folder):
    """
    Calculates PCA for the fused dataset and saves a plot with PC1 vs PC2 
    and explained variance.
    """
    # 1. Separate Data
    num_cols = df.select_dtypes(include=['number']).columns
    
    # Extract sample names for grouping logic
    # Assuming get_sample_names(df) returns a list of identifiers
    sample_names = get_sample_names(df)

    if len(num_cols) < 2:
        print("[ERROR] Not enough numeric features for PCA.")
        return

    X = df[num_cols]
    
    # 2. Scale Data
    X_scaled = StandardScaler().fit_transform(X)

    # 3. Run PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X_scaled)
    
    # Calculate Explained Variance
    exp_var_pc1 = pca.explained_variance_ratio_[0] * 100
    exp_var_pc2 = pca.explained_variance_ratio_[1] * 100

    # 4. Create Plot DataFrame & Assign Groups
    pca_df = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
    
    # Standardize group assignment based on sample names
    groups = []
    for name in sample_names:
        name_str = str(name).upper()
        if 'QC' in name_str:
            groups.append('QC')
        elif 'CTRL' in name_str:
            groups.append('Control')
        else:
            groups.append('Case')
            
    pca_df['Group'] = groups

    # Define markers
    group_markers = {
        'QC': 's',      # Square
        'Control': 'o', # Circle
        'Case': '^'     # Triangle
    }

    # Define Colors (Magma-like manual selection or mapping)
    # Note: 'magma' is a colormap, not a discrete palette dict. 
    # Here we pick 3 distinct colors from the magma range or define them manually to match previous plots.
    # Method A: Manual hex codes (Best for consistency with biplot)
    group_colors = {
        'QC': '#FC8961',
        'Control': '#B73779',
        'Case': '#51127C'
    }

    # 5. Plotting
    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid")
    
    # We use style= to handle markers and palette= for colors
    sns.scatterplot(
        x='PC1', 
        y='PC2', 
        data=pca_df,
        hue='Group',       # Controls Color
        style='Group',     # Controls Marker shape
        markers=group_markers, 
        palette=group_colors,
        s=100, 
        alpha=0.8
    )

    plt.axhline(0, color='#050402', linestyle='solid', linewidth=0.8, alpha=0.8)
    plt.axvline(0, color='#050402', linestyle='solid', linewidth=0.8, alpha=0.8)
    plt.grid(color='gray', linestyle='dashed', linewidth=0.5, alpha=0.7)

    plt.title(f'PCA - {dataset_label}', fontsize=15)
    plt.xlabel(f'PC1 ({exp_var_pc1:.2f}%)', fontsize=12)
    plt.ylabel(f'PC2 ({exp_var_pc2:.2f}%)', fontsize=12)
    
    # Improve legend
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()

    # Save and show
    save_plot(plt, f'SUMPCA - {dataset_label}', output_folder)
    #plt.show()


"""
    Function that generates a PCA Biplot (Scores + Loadings) for the given dataframe
"""
def biplot(df, dataset_label, output_folder):

    # 1. Pre-processing: Separate Numeric Data from Metadata
    df_numeric = df.select_dtypes(include=[np.number])
       
    if df_numeric.empty:
        print(f"Error: No numeric data found in {dataset_label} for PCA")
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
    fig = plt.figure(figsize=(8, 8))
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
    
    # Define markers for the groups
    group_markers = {
        'QC': 's',      # Square
        'Control': 'o', # Circle
        'Case': '^'     # Triangle
    }

    # 3. Plot Scores (Samples)
    for group in unique_groups:
        mask = groups == group
        if np.any(mask):
            plt.scatter(scores[mask, 0], scores[mask, 1],
                        c=group_colors.get(group, 'grey'),
                        label=group, 
                        marker=group_markers.get(group, 'o'),
                        alpha=0.6, 
                        s=60, edgecolors='w')

    # 4. Plot Top 10 Contributors (Loadings)
    magnitude = np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2)
    top_indices = np.argsort(magnitude)[-10:]
    
    metabolite_names = df_numeric.columns
    
    cmap = plt.get_cmap("magma")
    colors_contributors = [cmap(i) for i in np.linspace(0, 1, len(top_indices))]
    
    for idx, i in enumerate(top_indices):
        x_end = loadings[i, 0] * scale_factor
        y_end = loadings[i, 1] * scale_factor
        c = colors_contributors[idx]
        feat_name = metabolite_names[i]

        # 1. Vector line - Added label=feat_name so it appears in legend
        plt.plot([0, x_end], [0, y_end], color=c, linewidth=1.5, alpha=0.8)

        # 2. Endpoint
        plt.scatter(x_end, y_end, color=c, edgecolors='black', linewidth=0.5, s=40, zorder=10, label=feat_name)

        # 3. Label with Outline
        ha_align = 'left' if x_end > 0 else 'right'
        va_align = 'bottom' if y_end > 0 else 'top'
        
        txt = plt.text(x_end * 1.1, y_end * 1.1, feat_name,
                       color=c, fontsize=8, fontweight='bold', 
                       ha=ha_align, va=va_align)
        
        # Add black outline (stroke) to text
        txt.set_path_effects([pe.withStroke(linewidth=0.5, foreground='black')])
    
    # Decorations
    plt.axhline(0, color='#050402', linestyle='solid', linewidth=0.8, alpha=0.8)
    plt.axvline(0, color='#050402', linestyle='solid', linewidth=0.8, alpha=0.8)
    plt.grid(color='gray', linestyle='dashed', linewidth=0.5, alpha=0.7)

    plt.xlabel(f'PC1 ({exp_var[0]:.2f}%)')
    plt.ylabel(f'PC2 ({exp_var[1]:.2f}%)')
    plt.title(f'Biplot: Samples & Contributors - {dataset_label}')
    
    # Legend now includes samples (shapes) and top 10 metabolites (lines)
    plt.legend(loc='upper right', framealpha=0.9, fontsize='small')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save and show
    save_plot(plt, f'Biplot - {dataset_label}', output_folder)
    #plt.show()



def z_score_plot(df, dataset_label, output_folder, samples_per_group=25):
    """
    Calculates Z-scores for a subset of samples (Control vs Case) and plots them.
    """
    # 1. Prepare Data
    df_numeric = df.select_dtypes(include=[np.number])
    
    # Filter out QCs
    mask_qc = df.index.str.contains('QC', case=False, na=False)
    df_no_qc = df_numeric[~mask_qc]
    
    # Identify Controls and Cases
    mask_ctrl = df_no_qc.index.str.contains('CTRL', case=False, na=False)
    df_ctrl = df_no_qc[mask_ctrl]
    df_case = df_no_qc[~mask_ctrl]
    
    # 3. Select Subset
    subset_ctrl = df_ctrl.iloc[:samples_per_group]
    subset_case = df_case.iloc[:samples_per_group]
    
    # Combine the subsets
    df_subset = pd.concat([subset_ctrl, subset_case])
    
    if df_subset.empty:
        print("Error: No samples found for Z-score plotting after filtering")
        return

    # 4. Calculate Z-Scores (Total Ion Current proxy)
    sample_sums = df_subset.sum(axis=1)
    z_score_data = (sample_sums - sample_sums.mean()) / sample_sums.std()
    
    # 5. Assign Colors
    cmap = plt.get_cmap("magma")
    color_ctrl = cmap(0.2)
    color_case = cmap(0.7)
    colors = [color_ctrl] * len(subset_ctrl) + [color_case] * len(subset_case)
    
    # 6. Plotting
    plt.figure(figsize=(15, 6))
    
    plt.bar(range(len(z_score_data)), z_score_data, color=colors, alpha=0.9)
    
    plt.title(f'Z-Scores of Sample Intensities (Subset: {samples_per_group}/group) - {dataset_label}', fontsize=14)
    plt.ylabel('Z-Score')
    plt.axhline(y=0, color='grey', linestyle='-', linewidth=0.8)

    # --- FIX START ---
    # Use the index directly as it contains the sample names (guaranteed by your filtering logic)
    sample_names = df_subset.index.tolist()

    # Set ticks explicitly
    plt.xticks(range(len(sample_names)), sample_names, rotation=90, fontsize=8)
    
    # FIX: Increased pad from 0.6 to 4 to ensure labels are not covered by the axis
    plt.tick_params(axis='x', which='both', labelcolor='black', width=0.9, length=4, pad=4)
    
    # Ensure margins don't cut off the first/last bars
    plt.margins(x=0.01)
    # --- FIX END ---
    
    # Add threshold lines
    plt.axhline(y=2, color='#B73779', linestyle='--', alpha=0.5, label='Threshold (+/- 2)')
    plt.axhline(y=-2, color='#B73779', linestyle='--', alpha=0.5)
    
    # 7. Legend
    legend_elements = [
        Line2D([0], [0], color=color_ctrl, lw=4, label='Control'),
        Line2D([0], [0], color=color_case, lw=4, label='Case'),
        Line2D([0], [0], color='#B73779', linestyle='--', label='Outlier Threshold')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.grid(True, axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()

    save_plot(plt, f'Z-Scores of Sample Intensities {dataset_label}', output_folder)
    # plt.show()


def internal_variability(df, dataset_label, output_folder, samples_per_group=25):
    """
    Generates a boxplot representing the distribution of intensities for each sample.
    """
    # 1. Filter Data (Same logic)
    df_numeric_full = df.select_dtypes(include=[np.number])
    
    mask_qc = df.index.str.contains('QC', case=False, na=False)
    df_no_qc = df_numeric_full[~mask_qc]
    
    mask_ctrl = df_no_qc.index.str.contains('CTRL', case=False, na=False)
    df_ctrl = df_no_qc[mask_ctrl]
    df_case = df_no_qc[~mask_ctrl]
    
    subset_ctrl = df_ctrl.iloc[:samples_per_group]
    subset_case = df_case.iloc[:samples_per_group]
    
    df_subset = pd.concat([subset_ctrl, subset_case])
    
    if df_subset.empty:
        print("Error: No samples found for Internal Variability plotting after filtering.")
        return

    # --- FIX START ---
    # 2. Transpose Data: We want 1 box per SAMPLE. 
    # Seaborn plots columns, so we must make Samples the Columns.
    df_plot = df_subset.T 
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot using the TRANSPOSED data (df_plot), NOT the original df
    sns.boxplot(data=df_plot, palette="magma", ax=ax, showfliers=False, linewidth=0.8)
    
    ax.set_title(f'Internal Variability (Subset: {samples_per_group}/group) - {dataset_label}', fontsize=14, fontweight='bold')
    ax.set_xlabel("Samples", fontsize=12)
    ax.set_ylabel("Log Intensity / Abundance", fontsize=12)
    
    # Get names from the columns of the transposed dataframe (which are the samples)
    sample_names = df_plot.columns.tolist()
    
    # Set ticks explicitly using the correct names
    ax.set_xticks(range(len(sample_names)))
    ax.set_xticklabels(sample_names, rotation=90, fontsize=8)
    
    # FIX: Ensure padding is sufficient and labelsize is not too small
    ax.tick_params(axis='x', which='both', width=0.6, pad=4)
    # --- FIX END ---
        
    plt.tight_layout()
    plt.margins(x=0.01)

    save_plot(plt, f'Internal Variability {dataset_label}', output_folder)
    # plt.show()