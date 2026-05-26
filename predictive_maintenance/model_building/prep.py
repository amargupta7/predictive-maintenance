# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

hf_token = os.getenv("HF_TOKEN")

# Load the dataset directly from Hugging Face
api = HfApi(token=hf_token)
DATASET_PATH = "hf://datasets/amarg7/predictive_maintenance/engine_data.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Data Cleaning
# No need to Remove duplicates as we have already identified that we don't have any duplicates

# Handle Outliers (Capping Coolant temp based on EDA findings)
df['Coolant temp'] = df['Coolant temp'].clip(upper=100)

# Column Renaming for consistency
df.columns = [col.lower().replace(' ', '_') for col in df.columns]
print("Data cleaning and outlier handling complete.")

# Define Features (X) and Target (y)
X = df.drop(columns=['engine_condition'])
y = df['engine_condition']

# Perform 80/20 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Upload all 4 files back to Hugging Face
api = HfApi(token=hf_token)
X_train.to_csv("Xtrain.csv", index=False)
X_test.to_csv("Xtest.csv", index=False)
y_train.to_csv("ytrain.csv", index=False)
y_test.to_csv("ytest.csv", index=False)

# =========================
# Upload to Hugging Face
# =========================
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file in files:
    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=file,
        repo_id='amarg7/predictive_maintenance',
        repo_type="dataset",
    )

print("All 4 processed files uploaded to Hugging Face Hub successfully.")
