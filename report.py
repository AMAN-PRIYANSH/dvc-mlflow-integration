"""
report.py - builds report.html, the performance report, straight from MLflow.

It does NOT recompute anything. It asks the MLflow tracking server for every
run (metrics, confusion matrix, ROC curve points) and puts them into one
self-contained web page with:

    * a table of the DVC data versions (Git tag, commit, rows, md5 hash)
    * a picker: "Which data version would you like to see?"
    * accuracy, precision, recall, F1, specificity, ROC-AUC for every model
    * ROC curves and confusion matrices
    * a chart of how each score changes from v1.0 to v4.0

Open report.html in any browser (double-click it).

Usage:
    python report.py
"""
from __future__ import annotations

import json
import os
from datetime import datetime

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

import mlflow
import mlflow.artifacts
from mlflow.tracking import MlflowClient

import config

TEMPLATE = config.ROOT / "report_template.html"


def _latest_per_key(runs, key):
    """Keep only the newest run for each key (runs are already newest-first)."""
    seen, out = set(), []
    for r in runs:
        k = key(r)
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def collect() -> dict:
    """Read everything the report needs from the MLflow tracking server."""
    mlflow.set_tracking_uri(config.TRACKING_URI)
    client = MlflowClient()
    exp = client.get_experiment_by_name(config.EXPERIMENT_NAME)
    if exp is None:
        raise SystemExit("No MLflow experiment yet - run  python run_all.py  first.")

    def search(run_type):
        return client.search_runs([exp.experiment_id],
                                  filter_string=f"tags.run_type = '{run_type}'",
                                  order_by=["attributes.start_time DESC"], max_results=5000)

    parents = _latest_per_key(search("data_version"), lambda r: r.data.tags["data_version"])
    children = _latest_per_key(search("model"),
                               lambda r: (r.data.tags["data_version"], r.data.tags["model"]))

    versions = []
    for p in parents:
        prm = p.data.params
        versions.append({
            "version": prm["data_version"],
            "commit": prm["git_commit"],
            "train_rows": int(prm["train_rows"]),
            "new_rows": int(prm["new_rows_in_version"]),
            "test_rows": int(prm["test_rows"]),
            "malignant": int(prm["train_malignant"]),
            "benign": int(prm["train_benign"]),
            "train_md5": prm["dvc_train_md5"],
            "test_md5": prm["dvc_test_md5"],
            "parent_run_id": p.info.run_id,
        })
    versions.sort(key=lambda v: [int(x) for x in v["version"].lstrip("v").split(".")])

    runs = []
    for c in children:
        m = c.data.metrics
        roc = mlflow.artifacts.load_dict(f"{c.info.artifact_uri}/roc_curve.json")
        runs.append({
            "version": c.data.tags["data_version"],
            "model": c.data.tags["model"],
            "run_id": c.info.run_id,
            "metrics": {k: m[k] for k in config.METRICS},
            "cm": {k: int(m[f"cm_{k}"]) for k in ("tn", "fp", "fn", "tp")},
            "roc": roc,
        })

    models = [m for m in config.MODEL_COLORS if any(r["model"] == m for r in runs)]
    return {
        "meta": {
            "author": config.AUTHOR, "srn": config.SRN,
            "generated": datetime.now().strftime("%d %b %Y, %H:%M"),
            "experiment": config.EXPERIMENT_NAME,
            "tracking_uri": config.TRACKING_URI,
            "backend_store": config.BACKEND_STORE_URI,
        },
        "versions": versions,
        "models": models,
        "metrics": config.METRICS,
        "metric_labels": config.METRIC_LABELS,
        "runs": runs,
    }


def render(data: dict) -> tuple[str, str]:
    """Returns (full html page, page fragment without <html>/<head>/<body>)."""
    head, body = TEMPLATE.read_text(encoding="utf-8").split("<!--SPLIT-->")
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    body = body.replace("/*__DATA__*/", blob)
    full = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
            f"{head}</head>\n<body>\n{body}</body>\n</html>\n")
    return full, head + body


def build_report(out_path=None, fragment_path=None) -> str:
    data = collect()
    full, fragment = render(data)
    out_path = out_path or (config.ROOT / "report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full)
    if fragment_path:
        with open(fragment_path, "w", encoding="utf-8") as f:
            f.write(fragment)
    return str(out_path)


if __name__ == "__main__":
    print("report written to", build_report())
