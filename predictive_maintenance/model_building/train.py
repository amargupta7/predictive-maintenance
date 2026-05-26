import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score, confusion_matrix, ConfusionMatrixDisplay
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow
import mlflow.sklearn
# for model serialization
import joblib

# =========================
# Setup
# =========================
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)

api = HfApi(token=HF_TOKEN)
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("predictive-maintenance-prod")

# 1. Load the 4 files from Hugging Face
repo_id = "amarg7/predictive_maintenance"

X_train = pd.read_csv(f"hf://datasets/{repo_id}/Xtrain.csv")
X_test = pd.read_csv(f"hf://datasets/{repo_id}/Xtest.csv")
y_train = pd.read_csv(f"hf://datasets/{repo_id}/ytrain.csv").values.ravel()
y_test = pd.read_csv(f"hf://datasets/{repo_id}/ytest.csv").values.ravel()

print("HF data loaded")
# Preprocessing Setup (StandardScaler)
numeric_features = X_train.columns.tolist()
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features)
)

# Handle Class Imbalance
# Calculate weight: (Count of Negative / Count of Positive)
count_neg = (y_train == 0).sum()
count_pos = (y_train == 1).sum()
class_weight = count_neg / count_pos

# Define XGBoost Pipeline
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

model_pipeline = make_pipeline(preprocessor, xgb_model)

# Define Hyperparameter Grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 100],
    'xgbclassifier__max_depth': [3, 5],
    'xgbclassifier__learning_rate': [0.01, 0.1],
    'xgbclassifier__reg_lambda': [0.5, 1.0]
}

with mlflow.start_run(run_name="XGBoost_Pipeline_Optimized"):
    # Grid Search
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1, scoring='f1')
    grid_search.fit(X_train, y_train)

    # Nested MLflow Logging for every combination
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        with mlflow.start_run(run_name=f"set_{i}", nested=True):
            mlflow.log_params(results['params'][i])
            mlflow.log_metric("mean_test_f1", results['mean_test_score'][i])
            mlflow.log_metric("std_test_f1", results['std_test_score'][i])

    # Log best parameters in the main run
    mlflow.log_params(grid_search.best_params_)

    # 7. Evaluate with Custom Threshold (0.45)
    best_model = grid_search.best_estimator_
    threshold = 0.45

    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    report = classification_report(y_test, y_pred, output_dict=True)

    mlflow.log_metrics({
        "test_accuracy": report['accuracy'],
        "test_precision": report['1']['precision'],
        "test_recall": report['1']['recall'],
        "test_f1-score": report['1']['f1-score']
    })

    # Save and Register Model
    best_model_path = "best_predictive_model.joblib"
    joblib.dump(best_model, best_model_path)

    api = HfApi(token=HF_TOKEN)
    model_repo_id = f"{repo_id.split('/')[0]}/engine-condition-predictor"

    api.create_repo(repo_id=model_repo_id, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=best_model_path,
        path_in_repo=best_model_path,
        repo_id=model_repo_id,
        repo_type="model"
    )
    mlflow.sklearn.log_model(best_model, "model")
    print(f"Prod Model successfully registered at {model_repo_id}")
