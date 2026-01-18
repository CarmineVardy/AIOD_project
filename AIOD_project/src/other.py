import os

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.config_visualization import *


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

"""
    Function that separates numeric data from strings and text inside the dataframe
"""
def get_df_numeric(df):
    return df.select_dtypes(include=[np.number])


"""
    Calculates Z-scores for a subset of samples (Control vs Case) and plots them
"""
def z_score_plot(df, dataset_label, output_folder, samples_per_group=25):
    if df.empty:
        print(f"Error: No numeric data found in {dataset_label}")
        return

    df_numeric = get_df_numeric(df)

    # 1. Analyze ALL samples to establish the same Color/Marker mapping
    sample_names_all = get_sample_names(df)
    groups = []
    for name in sample_names_all:
        name_str = str(name).upper()
        if 'QC' in name_str:
            groups.append('QC')
        elif 'CTRL' in name_str:
            groups.append('Control')
        else:
            groups.append('Case')

    unique_groups_all = pd.unique(groups)

    markers_map = {cls: MARKERS[i % len(MARKERS)] for i, cls in enumerate(unique_groups_all)}
    colors_map = {cls: DISCRETE_COLORS[i % len(DISCRETE_COLORS)] for i, cls in enumerate(unique_groups_all)}

    # Identify Controls and Cases
    mask_ctrl = df_numeric.index.str.contains('CTRL', case=False, na=False)
    df_ctrl = df_numeric[mask_ctrl]
    df_case = df_numeric[~mask_ctrl]

    # 3. Select Subset
    subset_ctrl = df_ctrl.iloc[:samples_per_group]
    subset_case = df_case.iloc[:samples_per_group]

    # Combine the subsets
    df_subset = pd.concat([subset_ctrl, subset_case])

    # 4. Calculate Z-Scores (Total Ion Current proxy)
    sample_sums = df_subset.sum(axis=1)
    z_score_data = (sample_sums - sample_sums.mean()) / sample_sums.std()

    # 5. Assign Colors from Global Palette
    # We assume standard order: 0 = Control (usually), 1 = Case
    color_ctrl = colors_map.get('Control', 'blue')
    color_case = colors_map.get('Case', 'red')

    colors = [color_ctrl] * len(subset_ctrl) + [color_case] * len(subset_case)

    # 1. Flatten the dataframe to treat it as one giant list of numbers
    flat_data = df_numeric.values.flatten()

    # 2. Calculate the Single Global Mean and Standard Deviation
    global_mean = flat_data.mean()
    global_std = flat_data.std(ddof=1)  # ddof=1 for Sample Std Dev

    # 3. Calculate the Single Z-score for a specific point
    # Example: Let's calculate the Z-score for the MAXIMUM value in the dataset
    max_value = flat_data.max()
    max_zscore = (max_value - global_mean) / global_std

    print(f"Global Standard Deviation: {global_std:.4f}")
    print(f"Maximum Value in Data:     {max_value:.4f}")
    print(f"Z-Score of Maximum Value:  {max_zscore:.4f}")

    # 6. Plotting
    plt.figure(figsize=(15, 6))

    plt.bar(range(len(z_score_data)), z_score_data, color=colors, alpha=0.9, edgecolor='black', linewidth=0.5)

    plt.title(f'Z-Scores of Sample Intensities (Subset: {samples_per_group}/group) - {dataset_label}', fontsize=14,
              fontweight='bold')
    plt.ylabel('Z-Score')
    plt.axhline(y=0, color='grey', linestyle='-', linewidth=0.8)

    sample_names = df_subset.index.tolist()

    # Set ticks explicitly with rotation and smaller font
    plt.xticks(range(len(sample_names)), sample_names, rotation=70, fontsize=8, ha='center')

    # Ensure labels are not covered
    plt.tick_params(axis='x', which='both', labelcolor='black', width=0.9, length=4, pad=4)

    plt.margins(x=0.01)

    # Add threshold lines
    threshold_color = 'red'  # Distinct color for threshold
    plt.axhline(y=2, color=threshold_color, linestyle='--', alpha=0.5)
    plt.axhline(y=-2, color=threshold_color, linestyle='--', alpha=0.5)

    # 7. Legend
    legend_elements = [
        Line2D([0], [0], marker=markers_map.get('Control', 'o'), color='w',
               markerfacecolor=color_ctrl, markersize=10, markeredgecolor='k', label='Control'),

        Line2D([0], [0], marker=markers_map.get('Case', '^'), color='w',
               markerfacecolor=color_case, markersize=10, markeredgecolor='k', label='Case'),

        Line2D([0], [0], color=threshold_color, linestyle='dashed', label='Threshold (+/- 2)')
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
    df_numeric = get_df_numeric(df)

    if df_numeric.empty:
        print(f"Error: No numeric data found in {dataset_label}")
        return

    # 1. Analyze ALL samples to establish the same Color/Marker mapping
    sample_names_all = get_sample_names(df)
    groups = []
    for name in sample_names_all:
        name_str = str(name).upper()
        if 'QC' in name_str:
            groups.append('QC')
        elif 'CTRL' in name_str:
            groups.append('Control')
        else:
            groups.append('Case')

    unique_groups_all = pd.unique(groups)

    markers_map = {cls: MARKERS[i % len(MARKERS)] for i, cls in enumerate(unique_groups_all)}
    colors_map = {cls: DISCRETE_COLORS[i % len(DISCRETE_COLORS)] for i, cls in enumerate(unique_groups_all)}

    # Filter Controls and Cases (assuming 'df' index has sample names)
    mask_ctrl = df_numeric.index.str.contains('CTRL', case=False, na=False)
    df_ctrl = df_numeric[mask_ctrl]
    df_case = df_numeric[~mask_ctrl]

    subset_ctrl = df_ctrl.iloc[:samples_per_group]
    subset_case = df_case.iloc[:samples_per_group]

    # Combine (Samples as Rows)
    df_subset = pd.concat([subset_ctrl, subset_case])

    # Transpose for Boxplot (Samples on X-axis = Columns in Dataframe for plotting)
    df_plot = df_subset.T

    fig, ax = plt.subplots(figsize=(16, 8))

    # Create a palette list matching the columns (samples)
    color_ctrl = colors_map.get('Control', 'blue')
    color_case = colors_map.get('Case', 'red')

    palette_list = [color_ctrl] * len(subset_ctrl) + [color_case] * len(subset_case)

    # Plot using the TRANSPOSED data
    sns.boxplot(data=df_plot,
                palette=palette_list,
                ax=ax,
                showfliers=False,
                showmeans=False,
                meanline=False,
                linewidth=0.8,
                boxprops=dict(alpha=0.5),
                meanprops={'linestyle': 'dotted', 'linewidth': 2.0, 'color': 'black'},
                medianprops={'linestyle': 'solid', 'linewidth': 1.5, 'color': 'black'})

    # ax.set_yscale('log')
    ax.set_title(f'Internal Variability (Subset: {samples_per_group}/group) - {dataset_label}', fontsize=14,
                 fontweight='bold')
    ax.set_xlabel("Samples", fontsize=12)
    ax.set_ylabel("Abundance", fontsize=12)

    # Get names from the columns of the transposed dataframe
    sample_names = df_plot.columns.tolist()

    # Set ticks explicitly with rotation
    ax.set_xticks(range(len(sample_names)))
    ax.set_xticklabels(sample_names, rotation=70, fontsize=8, ha='right')

    # Adjust padding
    ax.tick_params(axis='x', which='both', width=0.6, pad=4)

    # 7. Legend
    legend_elements = [
        Line2D([0], [0], marker=markers_map.get('Control', 'o'), color='w',
               markerfacecolor=color_ctrl, markersize=10, markeredgecolor='k', label='Control'),

        Line2D([0], [0], marker=markers_map.get('Case', '^'), color='w',
               markerfacecolor=color_case, markersize=10, markeredgecolor='k', label='Case'),

        # Line Explanations
        Line2D([0], [0], color='black', linestyle='solid', linewidth=1.5, label='Median'),
        # Line2D([0], [0], color='black', linestyle='dashed', linewidth=1.5, label='Mean')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.margins(x=0.01)

    save_plot(plt, f'Internal Variability {dataset_label}', output_folder)
    # plt.show()