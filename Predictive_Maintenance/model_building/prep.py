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

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/debrupa24/predictive-maintenance-analysis/engine_data.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Identify target column

target_col = "Engine Condition"
X = df.drop(columns=[target_col])
y = df[target_col]


# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train.to_csv("Xtrain.csv",index=False)
X_test.to_csv("Xtest.csv",index=False)
y_train.to_csv("ytrain.csv",index=False)
y_test.to_csv("ytest.csv",index=False)



files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

# Upload to Huggingface
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id="debrupa24/predictive-maintenance-analysis",
        repo_type="dataset",
    )
