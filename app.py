"""
app.py - the GUI.  Start it with:   streamlit run app.py

What it does, in order:
    1. asks you: "Which data version would you like?"   (v1.0, v2.0, v3.0, v4.0)
    2. asks DVC for exactly that version of the data     (dvc.api.open at the Git tag)
    3. trains the models you tick                         (train.py)
    4. saves every run in MLflow                          (tracking server, SQLite backend)
    5. reads the runs back from MLflow and shows the performance:
       accuracy, precision, recall, F1, specificity, ROC-AUC, ROC curves,
       confusion matrices, and a comparison across all versions.

The GUI is the "frontend". MLflow is the "backend": the GUI never keeps results
itself - it always reads them back from the MLflow server.
"""
from __future__ import annotations

import math
import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

import altair as alt
import mlflow
import mlflow.artifacts
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from mlflow.tracking import MlflowClient

import config
import data_versions
import mlflow_server
import report
import train

st.set_page_config(page_title="DVC + MLflow model lab", layout="wide")

MODELS = list(config.MODEL_COLORS)
COLOR_SCALE = alt.Scale(domain=MODELS, range=[config.MODEL_COLORS[m] for m in MODELS])
LABELS = config.METRIC_LABELS


# ------------------------------------------------------------------ backend
@st.cache_resource(show_spinner="Starting the MLflow tracking server ...")
def start_backend() -> bool:
    mlflow_server.start()
    mlflow.set_tracking_uri(config.TRACKING_URI)
    return True


def experiment_id() -> str | None:
    exp = MlflowClient().get_experiment_by_name(config.EXPERIMENT_NAME)
    return exp.experiment_id if exp else None


def run_link(exp_id: str, run_id: str) -> str:
    return f"{config.TRACKING_URI}/#/experiments/{exp_id}/runs/{run_id}"


@st.cache_data(show_spinner=False)
def versions() -> list[str]:
    return data_versions.list_versions()


@st.cache_data(show_spinner="Asking DVC for this data version ...")
def version_data(v: str) -> dict:
    return data_versions.load_version(v)


@st.cache_data(show_spinner="Reading runs from the MLflow backend ...")
def mlflow_results(stamp: int) -> pd.DataFrame:
    """Latest run of every (data version, model) pair, read from MLflow.
    `stamp` changes after each training, which forces a fresh read."""
    exp_id = experiment_id()
    if exp_id is None:
        return pd.DataFrame()
    runs = MlflowClient().search_runs(
        [exp_id], filter_string="tags.run_type = 'model'",
        order_by=["attributes.start_time DESC"], max_results=5000)
    rows, seen = [], set()
    for r in runs:
        key = (r.data.tags["data_version"], r.data.tags["model"])
        if key in seen:
            continue
        seen.add(key)
        m = r.data.metrics
        rows.append({
            "version": key[0], "model": key[1], "run_id": r.info.run_id,
            "started": pd.to_datetime(r.info.start_time, unit="ms"),
            "link": run_link(exp_id, r.info.run_id),
            **{k: m.get(k) for k in config.METRICS},
            **{f"cm_{k}": int(m.get(f"cm_{k}", 0)) for k in ("tn", "fp", "fn", "tp")},
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def roc_points(run_id: str) -> dict:
    return mlflow.artifacts.load_dict(f"runs:/{run_id}/roc_curve.json")


@st.cache_data(show_spinner=False)
def artifact_file(run_id: str, path: str) -> str:
    return mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=path)


# ------------------------------------------------------------------ charts
def roc_chart(res: pd.DataFrame) -> alt.Chart:
    rows = []
    for _, r in res.iterrows():
        pts = roc_points(r["run_id"])
        label = f"{r['model']} (AUC {r['roc_auc']:.3f})"
        rows += [{"model": r["model"], "label": label, "fpr": f, "tpr": t}
                 for f, t in zip(pts["fpr"], pts["tpr"])]
    df = pd.DataFrame(rows)
    diag = alt.Chart(pd.DataFrame({"fpr": [0, 1], "tpr": [0, 1]})).mark_line(
        color="#c3c2b7", strokeWidth=1).encode(x="fpr", y="tpr")
    lines = alt.Chart(df).mark_line(strokeWidth=2, interpolate="linear").encode(
        x=alt.X("fpr:Q", title="False positive rate (1 - specificity)",
                scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("tpr:Q", title="True positive rate (recall)", scale=alt.Scale(domain=[0, 1], nice=False)),
        color=alt.Color("model:N", scale=COLOR_SCALE,
                        legend=alt.Legend(title=None, orient="bottom-right", labelLimit=220)),
        detail="label:N",
        tooltip=[alt.Tooltip("label:N", title="Model"),
                 alt.Tooltip("fpr:Q", title="False positive rate", format=".3f"),
                 alt.Tooltip("tpr:Q", title="Recall", format=".3f")],
    )
    points = lines.mark_point(size=40, filled=True, opacity=0)   # invisible, bigger hover targets
    return (diag + lines + points).properties(height=380)


def learning_chart(allres: pd.DataFrame, metric: str) -> alt.Chart:
    order = versions()
    lo = math.floor((allres[metric].min() - 0.01) * 50) / 50      # round down to a 0.02 step
    base = alt.Chart(allres).encode(
        x=alt.X("version:N", sort=order, title="Data version (each adds a new batch of rows)",
                axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f"{metric}:Q", title=LABELS[metric], scale=alt.Scale(domain=[max(0, lo), 1])),
        color=alt.Color("model:N", scale=COLOR_SCALE, legend=alt.Legend(title=None, orient="bottom")),
        tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("version:N", title="Version"),
                 alt.Tooltip(f"{metric}:Q", title=LABELS[metric], format=".3f")],
    )
    return (base.mark_line(strokeWidth=2) +
            base.mark_circle(size=90, opacity=1, stroke="white", strokeWidth=2)).properties(height=360)


def metrics_table(res: pd.DataFrame):
    table = res[["model", *config.METRICS, "link"]].rename(
        columns={"model": "Model", "link": "MLflow run", **LABELS}).set_index("Model")
    styled = table.style.format({LABELS[k]: "{:.3f}" for k in config.METRICS}).highlight_max(
        subset=[LABELS[k] for k in config.METRICS], props="font-weight:700; background-color:#eaf1fb")
    st.dataframe(styled, column_config={"MLflow run": st.column_config.LinkColumn(
        "MLflow run", display_text="open in MLflow")}, width="stretch")
    st.caption("Highlighted = best value in each column. Malignant is the positive class. "
               "All versions use the same fixed test set, so the numbers are comparable.")


def show_results(v: str, res: pd.DataFrame):
    best = res.sort_values(["f1", "roc_auc"], ascending=False).iloc[0]
    st.success(f"Best model on **{v}**: **{best['model']}** - F1-score {best['f1']:.3f}, "
               f"accuracy {best['accuracy']:.3f}, ROC-AUC {best['roc_auc']:.3f}")
    cols = st.columns(len(config.METRICS))
    for c, k in zip(cols, config.METRICS):
        c.metric(LABELS[k], f"{best[k]:.3f}", help=f"{best['model']} on {v}")

    st.markdown("#### Performance metrics")
    metrics_table(res)

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("#### ROC curves")
        st.altair_chart(roc_chart(res), width="stretch")
    with right:
        st.markdown("#### Confusion matrices")
        st.caption("These images are loaded from the MLflow artifact store.")
        grid = st.columns(2)
        for i, (_, r) in enumerate(res.iterrows()):
            with grid[i % 2]:
                st.image(artifact_file(r["run_id"], "plots/confusion_matrix.png"), width="stretch")


# ------------------------------------------------------------------ page
start_backend()
if "stamp" not in st.session_state:
    st.session_state.stamp = 0

st.title("Breast cancer classifier: DVC data versions + MLflow")
st.caption(f"{config.AUTHOR} · {config.SRN}   |   Pick a data version → DVC loads it → "
           "models train → MLflow saves the run → the results are read back from MLflow")

with st.sidebar:
    st.header("MLflow backend")
    if mlflow_server.is_running():
        st.markdown(f":green[●] Tracking server running at `{config.TRACKING_URI}`")
    else:
        st.markdown(":red[●] Tracking server is not reachable")
    st.markdown(f"**Backend store**: `{config.BACKEND_STORE_URI}`  \n"
                "(a SQLite database file: runs, parameters, metrics)")
    st.markdown(f"**Artifact store**: `{config.ARTIFACTS_DESTINATION}`  \n"
                "(plots, ROC points, saved models)")
    st.link_button("Open the MLflow UI", config.TRACKING_URI, width="stretch")
    st.divider()
    st.header("DVC data versions")
    for v in versions():
        st.markdown(f"- `{v}` → commit `{data_versions.commit_of(v)}`")
    st.caption("Each version is a Git tag. Git keeps the small .dvc receipt, "
               "DVC keeps the real data file.")

if not versions():
    st.error("No data versions found. Run  `python run_all.py`  once first.")
    st.stop()

allres = mlflow_results(st.session_state.stamp)
tab_train, tab_compare, tab_report = st.tabs(
    ["Train on a data version", "Compare all versions", "Report"])

# ---------------- tab 1
with tab_train:
    vs = versions()
    info = {v: version_data(v) for v in vs}
    v = st.radio("**Which data version would you like?**", vs, index=len(vs) - 1,
                 horizontal=True, key="version_choice",
                 captions=[f"{len(info[x]['train'])} training rows" for x in vs])
    d = info[v]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training rows", len(d["train"]))
    c2.metric("New rows in this version", len(d["new_rows"]))
    c3.metric("Malignant / benign", f"{int(d['train'][config.TARGET].sum())} / "
                                    f"{int((1 - d['train'][config.TARGET]).sum())}")
    c4.metric("Fixed test rows", len(d["test"]))
    st.caption(f"Git commit `{d['commit']}` · DVC md5 of train.csv `{d['train_md5']}`")

    with st.expander(f"See the {len(d['new_rows'])} rows that are new in {v}"):
        st.dataframe(d["new_rows"], hide_index=True, width="stretch", height=260)
    with st.expander("See the DVC receipt for this version (data/train.csv.dvc)"):
        st.code(d["train_receipt_text"], language="yaml")
        st.caption("Git stores only this small file. The md5 line points to the real data "
                   "inside DVC's cache.")

    chosen = st.multiselect("Models to train", MODELS, default=MODELS)
    if st.button(f"Train on {v} and log to MLflow", type="primary", disabled=not chosen):
        bar = st.progress(0.0, text="Starting ...")
        train.train_version(v, chosen, progress=lambda f, msg: bar.progress(min(f, 1.0), text=msg))
        st.session_state.stamp += 1
        mlflow_results.clear()
        allres = mlflow_results(st.session_state.stamp)
        st.toast(f"Saved {len(chosen)} runs for {v} in MLflow")

    res = allres[allres["version"] == v] if len(allres) else pd.DataFrame()
    st.divider()
    if res.empty:
        st.info(f"MLflow has no runs for {v} yet. Press the button above to train.")
    else:
        latest = res["started"].max().strftime("%d %b %Y %H:%M")
        st.subheader(f"Results for {v}")
        st.caption(f"Read from the MLflow backend (latest runs, {latest} UTC)")
        res = res.set_index("model").reindex([m for m in MODELS if m in set(res["model"])]).reset_index()
        show_results(v, res)

# ---------------- tab 2
with tab_compare:
    if allres.empty:
        st.info("No runs in MLflow yet.")
    else:
        metric = st.segmented_control("Metric", config.METRICS, default="f1",
                                      format_func=lambda k: LABELS[k], key="lc_metric") or "f1"
        st.altair_chart(learning_chart(allres, metric), width="stretch")
        st.caption("One point per data version, all tested on the same patients. "
                   "The vertical axis does not start at 0, so small differences are visible.")
        pivot = allres.pivot(index="model", columns="version", values=metric).rename_axis(
            index="Model", columns=None)
        pivot = pivot.reindex(index=[m for m in MODELS if m in pivot.index],
                              columns=[x for x in versions() if x in pivot.columns])
        st.markdown(f"#### {LABELS[metric]} for every model and version")
        st.dataframe(pivot.style.format("{:.3f}").highlight_max(axis=None,
                     props="font-weight:700; background-color:#eaf1fb"), width="stretch")

# ---------------- tab 3
with tab_report:
    st.markdown("The report is one HTML file built from the runs stored in MLflow. "
                "You can open it in any browser, or submit it.")
    if st.button("Build the report from MLflow"):
        with st.spinner("Reading every run from MLflow ..."):
            path = report.build_report()
        st.session_state.report_path = path
    path = st.session_state.get("report_path") or (
        str(config.ROOT / "report.html") if (config.ROOT / "report.html").exists() else None)
    if path:
        html = open(path, encoding="utf-8").read()
        st.download_button("Download report.html", html, file_name="report.html",
                           mime="text/html")
        components.html(html, height=1600, scrolling=True)
