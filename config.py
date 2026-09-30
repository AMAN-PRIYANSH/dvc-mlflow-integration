"""
config.py - every setting of the project in one place.

If you want to change something (port, number of data versions, your name
on the report), change it here and nowhere else.
"""
from pathlib import Path

# ---------------------------------------------------------------- who / what
AUTHOR = "Aman Priyansh"
SRN = "PES1UG24AM030"
PROJECT_TITLE = "Data versioning with DVC + experiment tracking with MLflow"

# ---------------------------------------------------------------- folders
ROOT = Path(__file__).resolve().parent          # the project folder itself
DATA_DIR = ROOT / "data"

# Paths *inside the repo* (DVC and Git want repo-relative paths)
TRAIN_CSV = "data/train.csv"    # grows with every version (v1 -> v4)
TEST_CSV = "data/test.csv"      # fixed "exam paper", same in every version

TARGET = "malignant"            # 1 = malignant tumour, 0 = benign
ID_COL = "patient_id"           # just an ID, never used for training

# ---------------------------------------------------------------- data versions
N_VERSIONS = 4                  # v1.0, v2.0, v3.0, v4.0
TEST_FRACTION = 0.20            # 20% of patients kept aside as the fixed test set
SEED = 42                       # same random numbers every time -> reproducible

# ---------------------------------------------------------------- MLflow
MLFLOW_HOST = "127.0.0.1"
MLFLOW_PORT = 5000
TRACKING_URI = f"http://{MLFLOW_HOST}:{MLFLOW_PORT}"   # where train.py sends runs
BACKEND_STORE_URI = "sqlite:///mlflow.db"             # backend store = SQLite database file
ARTIFACTS_DESTINATION = "./mlartifacts"               # artifact store = plots, models
EXPERIMENT_NAME = "My_First_ML_Project"          # same name as in the MLflow + DVC document

# DVC remote = storage folder inside the repo, so anyone who clones it can "dvc pull"
DVC_REMOTE_DIR = "dvc_storage"

# name of the saved model inside each MLflow run (the document uses "random_forest_model")
MODEL_ARTIFACT_NAMES = {
    "Logistic Regression": "logistic_regression_model",
    "Random Forest": "random_forest_model",
    "SVM (RBF kernel)": "svm_model",
    "K-Nearest Neighbours": "knn_model",
}

# ---------------------------------------------------------------- look
# One fixed colour per model (colour-blind-safe order), used in every chart.
MODEL_COLORS = {
    "Logistic Regression": "#2a78d6",   # blue
    "Random Forest": "#eb6834",         # orange
    "SVM (RBF kernel)": "#1baf7a",      # aqua
    "K-Nearest Neighbours": "#eda100",  # yellow
}

METRICS = ["accuracy", "precision", "recall", "f1", "specificity", "roc_auc"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
    "specificity": "Specificity",
    "roc_auc": "ROC-AUC",
}
