
import os
import pandas as pd

#Constant for PATHS
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

#Data Paths
DATA_PATH = os.path.join(PROJECT_PATH, "data")
DATA_RAW_PATH = os.path.join(DATA_PATH, 'raw')
NEG_RAW_DATASET_PATH = os.path.join(DATA_RAW_PATH, '2024_Metabolomica_Neg.csv')
POS_RAW_DATASET_PATH = os.path.join(DATA_RAW_PATH, '2024_Metabolomica_Pos.csv')


try:
    df_neg_raw = pd.read_csv(NEG_RAW_DATASET_PATH)
    df_pos_raw = pd.read_csv(POS_RAW_DATASET_PATH)

    print("SUCCESS: Datasets loaded successfully.")

except FileNotFoundError:
    print(f"ERROR: Files not found in {NEG_RAW_DATASET_PATH}. Please check the folder name and path.")