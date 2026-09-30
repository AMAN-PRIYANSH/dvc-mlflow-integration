"""
train.py - trains 4 models on ONE data version and logs everything to MLflow.

The chain:   DVC gives the data  ->  models train  ->  MLflow records the run

For the chosen version (e.g. v2.0) it creates, in MLflow:

    parent run  "data v2.0"                  <- which data: DVC md5 hash, Git commit,
    |                                           rows, the .dvc receipt, the dataset
    |-- child run "Logistic Regression"      <- settings (parameters), scores (metrics),
    |-- child run "Random Forest"               ROC curve, confusion matrix, saved model
    |-- child run "SVM (RBF kernel)"
    |-- child run "K-Nearest Neighbours"

Usage:
    python train.py --version v2.0
    python train.py --version all
"""
from __future__ import annotations

import argparse
import inspect
import os
import warnings

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import matplotlib
matplotlib.use("Agg")                     # draw plots without opening a window
import matplotlib.pyplot as plt
import mlflow
import mlflow.data
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config
import data_versions

# MLflow prints a long hint because the 0/1 prediction column is an integer. Harmless.
warnings.filterwarnings("ignore", message=".*Inferred schema contains integer column.*")


# ------------------------------------------------------------------ the 4 models
def build_models() -> dict:
    """name -> (model, the settings we want MLflow to remember)."""
    s = config.SEED
    return {
        "Logistic Regression": (
            make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
            {"C": 1.0, "max_iter": 2000, "scaling": "StandardScaler"},
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=200, max_depth=None, random_state=s),
            {"n_estimators": 200, "max_depth": "None", "random_state": s},
        ),
        "SVM (RBF kernel)": (
            make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf")),
            {"C": 1.0, "kernel": "rbf", "scaling": "StandardScaler"},
        ),
        "K-Nearest Neighbours": (
            make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
            {"n_neighbors": 5, "scaling": "StandardScaler"},
        ),
    }


# ------------------------------------------------------------------ the scores
def malignant_score(model, X):
    """How strongly the model believes each patient is malignant.

    Most models give a probability (0..1). The SVM gives a 'decision score'
    instead (distance from its boundary). ROC-AUC only needs the ORDER of the
    scores, so both work the same way for the ROC curve.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def compute_metrics(y_true, y_pred, y_score) -> tuple[dict, dict]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),                    # right / all
        "precision": precision_score(y_true, y_pred, zero_division=0), # of "malignant" calls, how many were right
        "recall": recall_score(y_true, y_pred, zero_division=0),       # of real malignant, how many we caught
        "f1": f1_score(y_true, y_pred, zero_division=0),               # balance of precision and recall
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,           # of real benign, how many we cleared
        "roc_auc": roc_auc_score(y_true, y_score),                     # ranking quality over all thresholds
    }
    cm = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return {k: float(v) for k, v in metrics.items()}, cm


# ------------------------------------------------------------------ the plots
def plot_confusion_matrix(cm: dict, title: str):
    grid = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8), dpi=120)
    ax.imshow(grid, cmap="Blues", vmin=0)
    for (i, j), v in np.ndenumerate(grid):
        dark = v > grid.max() * 0.55
        ax.text(j, i, str(v), ha="center", va="center", fontsize=15,
                color="white" if dark else "#0b0b0b")
    ax.set_xticks([0, 1], ["benign", "malignant"])
    ax.set_yticks([0, 1], ["benign", "malignant"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def plot_roc(fpr, tpr, auc: float, name: str, color: str):
    fig, ax = plt.subplots(figsize=(4.4, 4.0), dpi=120)
    ax.plot([0, 1], [0, 1], color="#c3c2b7", lw=1, label="random guess (AUC 0.50)")
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC {auc:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curve", fontsize=10)
    ax.grid(color="#e1e0d9", lw=0.8)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ one version
def log_dataset(df, path: str, version: str, md5: str, context: str) -> None:
    """Register the DVC data version as an MLflow dataset (name, digest, source)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")        # MLflow warns that the source is a plain path
        ds = mlflow.data.from_pandas(df, source=path, targets=config.TARGET,
                                     name=f"{path.split('/')[-1]}@{version}")
    mlflow.log_input(ds, context=context,
                     tags={"dvc_version": version, "dvc_md5": md5})


def log_sklearn_model(model, example, artifact_name: str = "model") -> None:
    """Save the trained model inside the MLflow run (so it can be reloaded later)."""
    kwargs = {"name": artifact_name, "input_example": example}
    if "skops_trusted_types" in inspect.signature(mlflow.sklearn.log_model).parameters:
        # Newer MLflow saves models with 'skops' and asks which object types to trust.
        # We trained this model ourselves a moment ago, so its own types are safe.
        import skops.io as sio
        kwargs["skops_trusted_types"] = sio.get_untrusted_types(data=sio.dumps(model))
    mlflow.sklearn.log_model(model, **kwargs)


def setup_mlflow() -> str:
    mlflow.set_tracking_uri(config.TRACKING_URI)
    exp = mlflow.set_experiment(config.EXPERIMENT_NAME)
    return exp.experiment_id


def train_version(version: str, model_names: list[str] | None = None,
                  progress=None) -> dict:
    """Train the chosen models on one DVC data version and log to MLflow.

    progress(fraction, message) is an optional callback (the GUI uses it).
    """
    say = progress or (lambda frac, msg: print(f"[{frac:4.0%}] {msg}"))
    experiment_id = setup_mlflow()

    # ---- 1. ask DVC for the data of this version
    say(0.02, f"Asking DVC for data version {version} ...")
    data = data_versions.load_version(version)
    train_df, test_df = data["train"], data["test"]
    drop = [config.ID_COL, config.TARGET]
    X_train, y_train = train_df.drop(columns=drop), train_df[config.TARGET]
    X_test, y_test = test_df.drop(columns=drop), test_df[config.TARGET]

    models = build_models()
    chosen = model_names or list(models)
    results = []

    # ---- 2. parent run = "this data version"
    with mlflow.start_run(run_name=f"data {version}") as parent:
        mlflow.set_tags({
            "run_type": "data_version",
            "data_version": version,
            "git_commit": data["commit"],
            "dvc_train_md5": data["train_md5"],
            "mlflow.note.content": (
                f"Models trained on DVC data version **{version}** "
                f"(Git commit {data['commit']}). train.csv md5 = {data['train_md5']}, "
                f"{len(train_df)} training rows, {len(test_df)} fixed test rows."),
        })
        mlflow.log_params({
            "dataset_version": data["train_md5"],   # the DVC hash, named as in the document
            "data_version": version,
            "git_commit": data["commit"],
            "dvc_train_md5": data["train_md5"],
            "dvc_test_md5": data["test_md5"],
            "train_rows": len(train_df),
            "new_rows_in_version": len(data["new_rows"]),
            "test_rows": len(test_df),
            "train_malignant": int(y_train.sum()),
            "train_benign": int((1 - y_train).sum()),
        })
        # "pass the dataset to MLflow": it shows up in the UI's Datasets column
        log_dataset(train_df, config.TRAIN_CSV, version, data["train_md5"], "training")
        log_dataset(test_df, config.TEST_CSV, version, data["test_md5"], "testing")
        # staple the DVC receipts to the MLflow run
        mlflow.log_text(data["train_receipt_text"], "dvc_receipts/train.csv.dvc")
        mlflow.log_text(data["test_receipt_text"], "dvc_receipts/test.csv.dvc")

        # ---- 3. one child run per model
        for i, name in enumerate(chosen):
            model, params = models[name]
            say(0.08 + 0.9 * i / len(chosen), f"Training {name} on {version} ...")
            with mlflow.start_run(run_name=name, nested=True) as child:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_score = malignant_score(model, X_test)
                metrics, cm = compute_metrics(y_test, y_pred, y_score)
                fpr, tpr, _ = roc_curve(y_test, y_score)

                mlflow.set_tags({"run_type": "model", "model": name,
                                 "data_version": version, "git_commit": data["commit"],
                                 "dvc_train_md5": data["train_md5"]})
                mlflow.log_params({"model": name,
                                   "dataset_version": data["train_md5"],   # DVC hash
                                   "data_version": version,
                                   "dvc_train_md5": data["train_md5"],
                                   "train_rows": len(train_df), **params})
                mlflow.log_metrics(metrics)
                mlflow.log_metrics({f"cm_{k}": v for k, v in cm.items()})
                mlflow.log_dict({"fpr": fpr.round(5).tolist(), "tpr": tpr.round(5).tolist()},
                                "roc_curve.json")
                mlflow.log_dict(cm, "confusion_matrix.json")
                mlflow.log_text(classification_report(
                    y_test, y_pred, target_names=["benign", "malignant"], digits=4),
                    "classification_report.txt")

                color = config.MODEL_COLORS.get(name, "#2a78d6")
                fig = plot_confusion_matrix(cm, f"{name} - data {version}")
                mlflow.log_figure(fig, "plots/confusion_matrix.png")
                plt.close(fig)
                fig = plot_roc(fpr, tpr, metrics["roc_auc"], name, color)
                mlflow.log_figure(fig, "plots/roc_curve.png")
                plt.close(fig)

                log_sklearn_model(model, X_test.head(3),
                                  config.MODEL_ARTIFACT_NAMES.get(name, "model"))

                results.append({"model": name, "run_id": child.info.run_id,
                                "metrics": metrics, "cm": cm,
                                "fpr": fpr.tolist(), "tpr": tpr.tolist()})

        # best model = highest F1-score (ties broken by ROC-AUC)
        best = max(results, key=lambda r: (r["metrics"]["f1"], r["metrics"]["roc_auc"]))
        mlflow.set_tag("best_model", best["model"])
        mlflow.log_metrics({f"best_{k}": v for k, v in best["metrics"].items()})

    say(1.0, f"Done: {len(results)} models trained on {version} and logged to MLflow")
    return {"version": version, "parent_run_id": parent.info.run_id,
            "experiment_id": experiment_id, "data": data, "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True, help='a Git tag like v2.0, or "all"')
    args = parser.parse_args()

    versions = data_versions.list_versions() if args.version == "all" else [args.version]
    for v in versions:
        out = train_version(v)
        for r in out["results"]:
            m = r["metrics"]
            print(f"  {v}  {r['model']:<22} acc={m['accuracy']:.3f} prec={m['precision']:.3f} "
                  f"rec={m['recall']:.3f} f1={m['f1']:.3f} spec={m['specificity']:.3f} "
                  f"auc={m['roc_auc']:.3f}")
