# Artificial Intelligence for Omics Data Analysis Project 2025-2026 
# Group 2 - Carmine Vardaro, Marco Savastano


The `main.py` script serves as the central executor for the entire project. It calls upon the various modules located in the `src/` directory to perform specific tasks.

It's important to note that this is not a rigid, sequential pipeline. While the current version of `main.py` executes a complete analysis workflow, it has also been used for temporary and exploratory analyses that may have been removed later. Therefore, `main.py` should be viewed as the primary entry point for running and orchestrating different parts of the project, with the final results and plots being saved to the `out/` directory.

## Project Structure

The project is organized as follows:

- **`main.py`**: The main script that executes the analysis.
- **`src/`**: Contains all the core Python modules.
  - **`analysis.py`**: Provides functions for dataset characterization, feature statistics, and univariate analysis.
  - **`anomalyDetection.py`**: Implements unsupervised anomaly detection algorithms like One-Class SVM, Isolation Forest, and Local Outlier Factor.
  - **`config_visualization.py`**: Contains global settings for all visualizations, ensuring a consistent style (color palettes, fonts, etc.).
  - **`dataFusion.py`**: Includes functions for low-level data fusion, combining data from different analytical modes.
  - **`data_loader.py`**: Handles loading of raw and processed datasets, as well as initial cleaning steps like removing QC samples.
  - **`datasetSplitting.py`**: Provides various strategies for splitting the data into training and test sets, including K-Fold, Stratified K-Fold, and Duplex.
  - **`models.py`**: Contains implementations of machine learning models (e.g., Logistic Regression, SVM, Random Forest) and functions for model evaluation.
  - **`pca.py`**: Dedicated to performing Principal Component Analysis (PCA) and generating related plots (scores, loadings, scree).
  - **`plsDa.py`**: Dedicated to performing Partial Least Squares Discriminant Analysis (PLS-DA) and its specific visualizations.
  - **`preprocessing.py`**: A collection of functions for data preprocessing, such as normalization, transformation, and scaling.
  - **`visualization.py`**: Contains functions for creating a wide range of plots, including box plots, density plots, and other custom visualizations.```
