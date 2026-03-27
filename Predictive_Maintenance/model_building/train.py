# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import  GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow
import subprocess
import mlflow
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

def evaluate_model(y_test, preds, probs):

    metrics = {}

    metrics["accuracy"] = accuracy_score(y_test, preds)
    metrics["precision"] = precision_score(y_test, preds)
    metrics["recall"] = recall_score(y_test, preds)
    metrics["f1_score"] = f1_score(y_test, preds)
    metrics["roc_auc"] = roc_auc_score(y_test, probs)

    return metrics



mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Predictive Maintenance V1")

api = HfApi()

login(token=os.getenv("HF_TOKEN"))
BASE = "https://huggingface.co/datasets/debrupa24/predictive-maintenance-analysis/resolve/main/"

X_train = pd.read_csv(BASE + "Xtrain.csv")
X_test = pd.read_csv(BASE + "Xtest.csv")
y_train = pd.read_csv(BASE + "ytrain.csv")
y_test = pd.read_csv(BASE + "ytest.csv")

print("Data loaded successfully")
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

gb_params = {
    "n_estimators": [100, 200],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5],
    "subsample": [0.8, 1]
}

gb = GradientBoostingClassifier(random_state=42)

gb_grid = RandomizedSearchCV(
    gb,
    gb_params,
    cv=5,
    scoring="recall",
    n_jobs=-1
)

with mlflow.start_run(run_name="GradientBoosting"):

    gb_grid.fit(X_train, y_train)

    best_gb = gb_grid.best_estimator_

    y_preds_gb = best_gb.predict(X_test)
    y_probs_gb = best_gb.predict_proba(X_test)[:,1]

    metrics = evaluate_model(y_test, y_preds_gb, y_probs_gb)

    mlflow.log_params(gb_grid.best_params_)

    for key, value in metrics.items():
        mlflow.log_metric(key, value)

    mlflow.sklearn.log_model(best_gb, "predictive_maintenance_gb_model")

    print("predictive maintenance GB Metrics:", metrics)

    # Save the model locally
    model_path = "/content/Predictive_Maintenance/model_building/predictive_maintenance_model_v1.joblib"
    joblib.dump(gb_grid, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "debrupa24/predictive_maintenance_model"
    repo_type = "model"

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")


    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="predictive_maintenance_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
        )
