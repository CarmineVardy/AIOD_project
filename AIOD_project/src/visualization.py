import os
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.stats import norm

# ==============================================================================
# GLOBAL VISUALIZATION SETTINGS
# ==============================================================================

# 1. GRAPHICS FORMAT CONFIGURATION
# Saving as PDF is preferred for high-resolution vector graphics in reports.
SAVE_FORMAT = 'pdf'

# Matplotlib global configuration for consistent styling
plt.rcParams.update({
    'figure.figsize': (10, 6),      # Default figure size
    'figure.dpi': 300,              # High resolution for display
    'axes.titlesize': 14,           # Title font size
    'axes.labelsize': 12,           # Axis label font size
    'xtick.labelsize': 10,          # X-axis tick font size
    'ytick.labelsize': 10,          # Y-axis tick font size
    'legend.fontsize': 10,          # Legend font size
    'lines.markersize': 8,          # Marker size for better visibility
    'font.family': 'sans-serif',    # Clean font family
    'pdf.fonttype': 42              # Ensure fonts are embedded as TrueType (editable in vector soft)
})

# ==============================================================================
# COLOR PALETTE & ACCESSIBILITY
# ==============================================================================

# AVAILABLE PALETTES (Viridis family - Colorblind friendly & Perceptually Uniform)
# Change the 'SELECTED_PALETTE' variable below to switch the theme for all plots.
# Options:
#   'viridis'  : The default. Blue -> Green -> Yellow. High contrast.
#   'plasma'   : Blue -> Red -> Yellow. Higher contrast, very vibrant.
#   'inferno'  : Black -> Red -> Yellow. Good for dark backgrounds or high intensity.
#   'magma'    : Black -> Purple -> Peach. Similar to inferno but softer.
#   'cividis'  : Specifically designed for color vision deficiency (CVD). Blue -> Yellow.
#   'mako'     : Dark Blue -> Green. Ocean-like.
#   'rocket'   : Dark Purple -> Red -> White.
#   'turbo'    : Rainbow alternative (use with caution, but better than Jet).

SELECTED_PALETTE = 'magma'  # <--- MODIFY THIS to test different variants

# Generate a list of discrete colors from the continuous colormap.
# For binary classes (CTRL vs CHD), the first two or specific indices will be used.
_cmap =  plt.get_cmap(SELECTED_PALETTE)
contrast_indices = [0.0, 0.5, 0.95, 0.25, 0.75, 0.15, 0.60, 0.35, 0.85]
DISCRETE_COLORS = [_cmap(i) for i in contrast_indices]

#DISCRETE_COLORS = [_cmap(i) for i in np.linspace(0, 1, 10)]

# Set Seaborn default palette to match
sns.set_palette(DISCRETE_COLORS)

# ==============================================================================
# MARKERS FOR BLACK & WHITE COMPATIBILITY
# ==============================================================================
# To ensure accessibility when printed in grayscale, we map classes to specific markers.
# Order: Circle, Square, Triangle Up, Diamond, Triangle Down, Cross, Plus
MARKERS = ['o', 's', '^', 'D', 'v', 'X', 'P']

# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================

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


def plot_volcano(results_df, output_dir, file_name="volcano_plot",
                 p_thresh=0.05, fc_thresh=2.0, use_adj_pval=True, top_n_labels=10):
    """
    Generates a Volcano Plot to visualize Univariate Analysis results.

    THEORY & IMPLEMENTATION DETAILS:
    --------------------------------
    1. Axes Definition:
       - X-axis: Magnitude of change (Log2 Fold Change).
         Positive = Up-regulated in Group 1 vs Group 2.
         Negative = Down-regulated.
       - Y-axis: Statistical Significance (-Log10 p-value).
         Higher values = More significant.
    2. Thresholds (Regions of Interest):
       - Horizontal Line: Statistical significance cutoff (e.g., p < 0.05).
         Can use raw p-value or FDR-adjusted p-value (recommended).
       - Vertical Lines: Biological relevance cutoff (Fold Change).
         Input is linear FC (e.g., 2.0), converted to +/- Log2(FC) for plotting.
    3. Quadrants:
       - Top-Right: Significantly Up-regulated.
       - Top-Left: Significantly Down-regulated.
       - Bottom / Center: Not significant (Noise).

    Args:
        results_df (pd.DataFrame): Output from 'perform_univariate_analysis'.
                                   Must contain 'p_value' (or 'p_value_adj') and 'log2_fc'.
        output_dir (str): Output directory.
        file_name (str): Filename.
        p_thresh (float): Significance threshold (default 0.05).
        fc_thresh (float): Fold Change threshold (linear, e.g., 2.0 implies 2x change).
        use_adj_pval (bool): If True, uses 'p_value_adj' (FDR). If False, uses raw 'p_value'.
        top_n_labels (int): Number of top significant features to annotate with text.
    """

    os.makedirs(output_dir, exist_ok=True)

    # 1. Prepare Data for Plotting
    df = results_df.copy()

    # Select which p-value to use
    pval_col = 'p_value_adj' if use_adj_pval else 'p_value'

    # Calculate -Log10(p-value) for Y-axis
    # Adding a tiny epsilon to avoid log(0) if p-value is extremely small
    df['neg_log10_p'] = -np.log10(df[pval_col] + 1e-300)

    # Calculate Log2 threshold from linear FC input
    # e.g., FC=2.0 -> log2(2)=1.0. We check for > 1.0 and < -1.0
    log2_fc_thresh = np.log2(fc_thresh)
    neg_log10_p_thresh = -np.log10(p_thresh)

    # 2. Categorize Points (Up, Down, NS)
    # Define conditions
    cond_sig = df[pval_col] < p_thresh
    cond_up = df['log2_fc'] > log2_fc_thresh
    cond_down = df['log2_fc'] < -log2_fc_thresh

    conditions = [
        (cond_sig & cond_up),
        (cond_sig & cond_down)
    ]
    choices = ['UP', 'DOWN']
    df['status'] = np.select(conditions, choices, default='NS')

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(10, 8))

    # Map colors: NS -> Grey, UP -> Red/Yellow-ish, DOWN -> Blue/Purple-ish
    # We pick colors from our global DISCRETE_COLORS or define specific ones for contrast
    # Using 'tab10' or manual mapping for clear distinction
    palette = {
        'NS': 'lightgrey',
        'UP': '#d62728',  # Red
        'DOWN': '#1f77b4'  # Blue
    }

    sns.scatterplot(data=df, x='log2_fc', y='neg_log10_p',
                    hue='status', style='status',
                    palette=palette,
                    hue_order=['NS', 'UP', 'DOWN'],
                    markers={'NS': 'o', 'UP': '^', 'DOWN': 'v'},
                    s=60, alpha=0.8, edgecolor='k', linewidth=0.5,
                    ax=ax)

    # 4. Add Threshold Lines
    # Horizontal (Significance)
    ax.axhline(neg_log10_p_thresh, color='k', linestyle='--', linewidth=1, alpha=0.7)
    # Vertical (Fold Change)
    ax.axvline(log2_fc_thresh, color='k', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(-log2_fc_thresh, color='k', linestyle='--', linewidth=1, alpha=0.7)

    # 5. Annotate Top Features
    # We sort by p-value (ascending) to find the most significant ones
    top_hits = df[df['status'] != 'NS'].sort_values(by=pval_col).head(top_n_labels)

    texts = []
    for idx, row in top_hits.iterrows():
        texts.append(plt.text(row['log2_fc'], row['neg_log10_p'], idx, fontsize=9))

    # Adjust text positions to avoid overlap (requires 'adjustText' library if available,
    # otherwise basic placement. Here we assume standard matplotlib).
    # If you have adjustText installed: from adjustText import adjust_text; adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))

    # 6. Labels and Title
    pval_label = "FDR (adj. p-value)" if use_adj_pval else "Raw p-value"
    ax.set_xlabel(r"$Log_2$ Fold Change", fontweight='bold')
    ax.set_ylabel(r"$-Log_{10}$ " + f"({pval_label})", fontweight='bold')

    title_str = f"Volcano Plot (Thresh: p<{p_thresh}, FC>{fc_thresh})"
    ax.set_title(title_str, fontweight='bold')

    # Add informative legend text about quadrants
    info_text = (f"UP: p<{p_thresh} & FC>{fc_thresh}\n"
                 f"DOWN: p<{p_thresh} & FC<1/{fc_thresh}")
    plt.text(0.98, 0.02, info_text, transform=ax.transAxes,
             fontsize=9, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.9))

    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{file_name}.{SAVE_FORMAT}")
    plt.savefig(save_path, format=SAVE_FORMAT, bbox_inches='tight')
    plt.close(fig)

    print(f"Generating Volcano Plot saved to {save_path}")
    print(f"   Significant features highlighted: {len(df[df['status'] != 'NS'])}")




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

    fig, ax1 = plt.subplots(figsize=(10, 6))

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

    plt.title('PCA Scree Plot (Explained Variance)', fontweight='bold')

    save_path = os.path.join(output_dir, f"{file_name}.{SAVE_FORMAT}")
    plt.savefig(save_path, format=SAVE_FORMAT, bbox_inches='tight')
    plt.close(fig)
    print(f"Generating PCA Scree Plot saved to {save_path}")


def plot_pca_scores(pca_results, df, output_dir, pc_x=1, pc_y=2, file_name="pca_score_plot", class_col='Class',
                    show_ellipse=True):
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

    # Map markers
    unique_classes = df[class_col].unique()
    current_palette = DISCRETE_COLORS[:len(unique_classes)]
    markers_dict = {cls: MARKERS[i % len(MARKERS)] for i, cls in enumerate(unique_classes)}

    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter Plot
    sns.scatterplot(x=x_data, y=y_data,
                    hue=df[class_col], style=df[class_col],
                    palette=current_palette, markers=markers_dict,
                    s=100, alpha=0.9, edgecolor='k', ax=ax)

    # --- HOTELLING'S T2 ELLIPSE CALCULATION ---
    if show_ellipse:
        # The ellipse is calculated on the assumption of multivariate normality of the SCORES.
        # Since PCA scores are orthogonal (uncorrelated), the axes of the ellipse align with X and Y.
        # Width/Height are proportional to the square root of the eigenvalues (variance) * Chi-Square critical value.

        # Chi-Square critical value for 95% confidence with 2 Degrees of Freedom is ~5.991
        chi2_val = 5.991

        # Variance of the plotted PCs
        std_x = np.std(x_data)
        std_y = np.std(y_data)

        # Width and Height (Total length, so 2 * radius)
        width = 2 * np.sqrt(chi2_val) * std_x
        height = 2 * np.sqrt(chi2_val) * std_y

        ellipse = Ellipse((0, 0), width=width, height=height,
                          facecolor='none', edgecolor='black', linestyle='--', linewidth=1.5,
                          label='95% Confidence (Hotelling T²)')
        ax.add_patch(ellipse)

    # Labels with Explained Variance
    var_x = var_ratio[pc_x - 1] * 100
    var_y = var_ratio[pc_y - 1] * 100

    ax.set_xlabel(f"{pc_x_label} ({var_x:.1f}%)", fontweight='bold')
    ax.set_ylabel(f"{pc_y_label} ({var_y:.1f}%)", fontweight='bold')
    ax.set_title(f"PCA Score Plot: {pc_x_label} vs {pc_y_label}", fontweight='bold')

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

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot all features
    # Usiamo un colore secondario della palette (es. ind. 3) per lo sfondo, con trasparenza
    ax.scatter(loadings[pc_x_label], loadings[pc_y_label], color=DISCRETE_COLORS[3], alpha=0.4, s=20, label='All Features')

    # Plot top features as highlighted points
    ax.scatter(top_loadings[pc_x_label], top_loadings[pc_y_label], color=DISCRETE_COLORS[0], s=50,
               label=f'Top {top_n} Contributors')

    # Annotate Top N
    texts = []
    for feature, row in top_loadings.iterrows():
        # Anche il testo usa il colore della palette
        texts.append(plt.text(row[pc_x_label], row[pc_y_label], feature, fontsize=8, color=DISCRETE_COLORS[0],
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



#DA RIVEDERE
def plot_loading_profile(pca_results, output_dir, pc_index=1, file_name="loading_profile", top_n=15):
    """
    Generates a 'Manhattan-style' loading plot for a SINGLE component.
    Useful to identify exactly which metabolites drive the separation on PCx.
    """
    os.makedirs(output_dir, exist_ok=True)

    pc_label = f"PC{pc_index}"
    loadings = pca_results['loadings'][pc_label].copy()

    # Sort by absolute value to find top contributors
    loadings_abs = loadings.abs().sort_values(ascending=False)
    top_features = loadings_abs.head(top_n).index

    # Filter data for plotting (we plot only top N for clarity, or all if preferred)
    # Here we visualize top N specifically to read names
    subset = loadings.loc[top_features]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create a stem plot (lollipop chart)
    # We use a color from our palette (e.g., color 0 for consistency)
    marker_color = DISCRETE_COLORS[0]

    (markers, stemlines, baseline) = ax.stem(range(len(subset)), subset.values)
    plt.setp(markers, marker='D', markersize=8, markeredgecolor='k', color=marker_color)
    plt.setp(stemlines, color=marker_color, linewidth=1.5)
    plt.setp(baseline, color='gray', linewidth=0.5)

    # Customizing X-axis
    ax.set_xticks(range(len(subset)))
    ax.set_xticklabels(subset.index, rotation=45, ha='right', fontsize=9, fontweight='bold')

    ax.set_ylabel(f"Loading Value ({pc_label})")
    ax.set_title(f"Top {top_n} Contributors to {pc_label}", fontweight='bold')
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{file_name}_{pc_label}.{SAVE_FORMAT}")
    plt.savefig(save_path, format=SAVE_FORMAT, bbox_inches='tight')
    plt.close(fig)
    print(f"Generating PCA Loading Profile ({pc_label}) saved to {save_path}")


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
        #plt.close()
        print(f"Graph {filename} saved in: {output_folder}")
    except:
        print(f"[ERROR] in saving {filename}: {e}")



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



"""
    Function that generates a PCA Biplot (Scores + Loadings) for the given dataframe
"""
def biplot(df, dataset_label, output_folder):

    # 1. Pre-processing: Separate Numeric Data from Metadata
    # The error 'could not convert string to float' happens because we are trying to scale text columns.
    df_numeric = df.select_dtypes(include=[np.number])
    
    # If the dataframe is empty after filtering, we cannot proceed
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

        # 1. Vector line
        # Removed invalid arguments: edgecolors, facecolors
        # Fixed duplicate linewidth (kept 1.5)
        plt.plot([0, x_end], [0, y_end], color=c, linewidth=1.5, alpha=0.8)

        # 2. Endpoint
        # Removed facecolors='none' so the dot is filled with color 'c'
        # Added edgecolors='black' for the outline
        plt.scatter(x_end, y_end, color=c, edgecolors='black', linewidth=0.5, s=40, zorder=10)

        # 3. Label
        # Added dynamic alignment so labels don't overlap the line
        ha_align = 'left' if x_end > 0 else 'right'
        va_align = 'bottom' if y_end > 0 else 'top'
        
        plt.text(x_end * 1.1, y_end * 1.1, feat_name,
                color=c, fontsize=8, fontweight='bold', 
                ha=ha_align, va=va_align)
    
    # Decorations
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='grey', linestyle='--', linewidth=0.8)
    plt.xlabel(f'PC1 ({exp_var[0]:.2f}%)')
    plt.ylabel(f'PC2 ({exp_var[1]:.2f}%)')
    plt.title(f'Biplot: Samples & Contributors - {dataset_label}')
    plt.legend(loc='upper right', framealpha=0.9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    

    # Save and show
    save_plot(plt, f'Biplot - {dataset_label}', output_folder)
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
        print("Error: No samples found for Z-score plotting after filtering")
        return

    # 4. Calculate Z-Scores
    # Calculate Sum of intensities per sample (Total Ion Current proxy)
    sample_sums = df_subset.sum(axis=1)
    
    # Calculate Z-Score relative to this subset's mean/std
    z_score_data = (sample_sums - sample_sums.mean()) / sample_sums.std()
    
    # 5. Assign Colors using Magma Palette

    cmap = plt.get_cmap("magma")
    color_ctrl = cmap(0.2) # dark purple/black
    color_case = cmap(0.7) # reddish/orange
    
    # Create a color list corresponding to the rows in df_subset
    # Since we concatenated [ctrl, case], the first N are ctrl, the rest are case.
    colors = [color_ctrl] * len(subset_ctrl) + [color_case] * len(subset_case)
    
    # 6. Plotting
    plt.figure(figsize=(15, 6))
    
    bars = plt.bar(range(len(z_score_data)), z_score_data, color=colors, alpha=0.9)
    
    plt.title(f'Z-Scores of Sample Intensities (Subset: {samples_per_group}/group) - {dataset_label}', fontsize=14)
    plt.ylabel('Z-Score')
    plt.axhline(y=0, color='grey', linestyle='-', linewidth=0.8)
    

    # Manage x-tick density using labels from the helper function
    sample_names = get_sample_names(df_subset)


    # Set X-ticks with sample names
    plt.xticks(range(len(sample_names)), df_subset.index, rotation=70, fontsize=8)
    # Increase space on X axis values
    plt.tick_params(axis='x', which='both', labelsize=5, width=0.6, pad=0.6)
    plt.margins(x=0.01)
    
    # Add threshold lines
    plt.axhline(y=2, color='#B73779', linestyle='--', alpha=0.5, label='Threshold (+/- 2)')
    plt.axhline(y=-2, color='#B73779', linestyle='--', alpha=0.5)
    
    # 7. Create Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=color_ctrl, lw=4, label='Control'),
        Line2D([0], [0], color=color_case, lw=4, label='Case'),
        Line2D([0], [0], color='#B73779', linestyle='--', label='Outlier Threshold')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.grid(True, axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()


    # Save and show
    save_plot(plt, f'Z-Scores of Sample Intensities {dataset_label}', output_folder)
    plt.show()



"""
    Generates a boxplot representing the distribution of intensities for each sample.
    Filters for a subset of samples (Control vs Case) similar to the Z-score plot.
"""
def internal_variability(df, dataset_label, output_folder, samples_per_group=25):
    
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
    #df_plot = df_subset.T 
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Generate Boxplot
    sns.boxplot(data=df, palette="magma", ax=ax, showfliers=False, linewidth=0.8)
    
    # Aesthetics
    ax.set_title(f'Internal Variability (Subset: {samples_per_group}/group) - {dataset_label}', fontsize=14, fontweight='bold')
    ax.set_xlabel("Samples", fontsize=12)
    ax.set_ylabel("Log Intensity / Abundance", fontsize=12)
    
    # Manage x-tick density using labels from the helper function
    sample_names = get_sample_names(df_subset)
    
    if df.shape[1] > 60:
        ax.set_xticks([]) # Hide labels if too crowded
    else:
        # Set ticks manually to ensure alignment with sample names
        ax.set_xticks(range(len(sample_names)))
        ax.set_xticklabels(sample_names, rotation=70, fontsize=8)
        
    plt.tight_layout()

    # Increase space on X axis values
    plt.tick_params(axis='x', which='both', labelsize=5, width=0.6, pad=0.6)
    plt.margins(x=0.01)

    # Save and show
    save_plot(plt, f'Internal Variability {dataset_label}', output_folder)
    plt.show()