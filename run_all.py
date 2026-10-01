"""
run_all.py - does the WHOLE project in one go.

    Step 1  Git + DVC set-up                       (git init, dvc init)
    Step 2  make data v1.0 ... v4.0                (for each: dvc add -> git commit -> git tag)
    Step 3  start the MLflow tracking server       (SQLite backend store)
    Step 4  train 4 models on every version        (logged to MLflow)
    Step 5  build report.html from MLflow          (the performance report)
    Step 6  show the "double checkout"             (go back to v1.0, then forward again)

Usage:
    python run_all.py            normal run (skips anything already done)
    python run_all.py --fresh    wipe Git/DVC/MLflow state and rebuild everything
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import config

GIT_EMAIL_FALLBACK = "amanbroken73@gmail.com"   # only used if Git has no email set yet

COMMIT_TRAILER = ""

CODE_FILES = ["config.py", "prepare_data.py", "data_versions.py", "mlflow_server.py",
              "train.py", "report.py", "report_template.html", "app.py", "run_all.py",
              "requirements.txt",
              "README.md", ".gitignore", ".gitattributes", ".streamlit/config.toml",
              "start_gui.bat", "setup.bat"]


def step(n, text):
    print(f"\n=== Step {n}: {text} " + "=" * max(0, 50 - len(text)), flush=True)


def run(cmd: list[str], quiet=False) -> str:
    shown = " ".join(cmd).replace(f"{sys.executable} -m ", "")   # show "dvc add ..." not the python path
    if not quiet:
        print(f"  $ {shown}", flush=True)
    out = subprocess.run(cmd, cwd=config.ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stdout, out.stderr)
        raise SystemExit(f"Command failed: {shown}")
    return out.stdout.strip()


def git(*args, quiet=False):
    return run(["git", *args], quiet=quiet)


def dvc(*args, quiet=False):
    return run([sys.executable, "-m", "dvc", *args], quiet=quiet)


def commit(message: str):
    git("commit", "-q", "-m", message + COMMIT_TRAILER, quiet=True)
    print(f'  $ git commit -m "{message}"')


# --------------------------------------------------------------------- step 0
def wipe():
    import mlflow_server
    if mlflow_server.is_running():
        raise SystemExit("MLflow server is running. Close it (and the GUI) before --fresh.")
    for p in [".git", ".dvc", ".dvcignore", "data", config.DVC_REMOTE_DIR, "mlflow.db", "mlartifacts", "mlruns",
              "report.html", "mlflow_server.log"]:
        path = config.ROOT / p
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()
    print("  wiped old Git / DVC / MLflow state")


# --------------------------------------------------------------------- step 1
def setup_repo():
    if (config.ROOT / ".git").exists() and (config.ROOT / ".dvc").exists():
        print("  Git and DVC already set up - skipping")
        return
    git("init", "-q", "-b", "main")
    if not _has_git_config():
        git("config", "user.name", config.AUTHOR, quiet=True)
        git("config", "user.email", GIT_EMAIL_FALLBACK, quiet=True)
    dvc("init", "-q")
    dvc("config", "core.analytics", "false", quiet=True)
    dvc("remote", "add", "-d", "storage", config.DVC_REMOTE_DIR)   # DVC remote = storage folder
    git("add", *[f for f in CODE_FILES if (config.ROOT / f).exists()], ".dvc", ".dvcignore")
    commit("Project setup: code + DVC initialised")


def _has_git_config() -> bool:
    r = subprocess.run(["git", "config", "--local", "user.email"], cwd=config.ROOT,
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


# --------------------------------------------------------------------- step 2
def make_versions():
    import data_versions
    import prepare_data

    existing = data_versions.list_versions()
    for k in range(1, config.N_VERSIONS + 1):
        tag = f"v{k}.0"
        if tag in existing:
            print(f"  {tag} already exists - skipping")
            continue
        print(f"\n  -- data version {tag}")
        prepare_data.write_version(k)
        dvc("add", "-q", config.TRAIN_CSV, config.TEST_CSV)
        dvc("push", "-q")                       # copy this version into the DVC remote
        git("add", f"{config.TRAIN_CSV}.dvc", f"{config.TEST_CSV}.dvc", "data/.gitignore",
            config.DVC_REMOTE_DIR)
        n_rows = sum(1 for _ in open(config.ROOT / config.TRAIN_CSV, encoding="utf-8")) - 1
        msg = f"Data {tag}: train.csv now has {n_rows} rows"
        commit(msg)
        git("tag", "-a", tag, "-m", msg, quiet=True)
        print(f"  $ git tag -a {tag}")


# --------------------------------------------------------------------- step 4
def train_all(retrain: bool):
    import mlflow
    import data_versions
    import train

    train.setup_mlflow()
    for v in data_versions.list_versions():
        done = mlflow.search_runs(
            filter_string=f"tags.run_type = 'data_version' and tags.data_version = '{v}'")
        if len(done) and not retrain:
            print(f"  {v}: already in MLflow - skipping (use --retrain to train again)")
            continue
        out = train.train_version(v)
        for r in out["results"]:
            m = r["metrics"]
            print(f"     {r['model']:<22} acc={m['accuracy']:.3f}  f1={m['f1']:.3f}  "
                  f"auc={m['roc_auc']:.3f}")


# --------------------------------------------------------------------- step 6
def checkout_demo():
    def rows():
        return sum(1 for _ in open(config.ROOT / config.TRAIN_CSV, encoding="utf-8")) - 1

    first, last = "v1.0", f"v{config.N_VERSIONS}.0"
    print(f"  now: data/train.csv has {rows()} rows")
    git("checkout", "-q", first)
    dvc("checkout", "-q")
    print(f"  after going back to {first}: data/train.csv has {rows()} rows")
    git("checkout", "-q", "main")
    dvc("checkout", "-q")
    print(f"  after coming back to main ({last}): data/train.csv has {rows()} rows")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fresh", action="store_true", help="wipe and rebuild everything")
    ap.add_argument("--retrain", action="store_true", help="train again even if runs exist")
    ap.add_argument("--stop-server", action="store_true", help="stop MLflow at the end")
    args = ap.parse_args()

    if shutil.which("git") is None:
        raise SystemExit("Git is not installed (or not on PATH). Install it from git-scm.com")

    if args.fresh:
        step(0, "wipe old state")
        wipe()

    step(1, "Git + DVC set-up")
    setup_repo()
    import data_versions
    if data_versions.ensure_local_remote():   # lets DVC find dvc_storage/ for old tags too
        print(f"  $ dvc remote modify --local storage url {config.DVC_REMOTE_DIR}")

    step(2, f"make data versions v1.0 ... v{config.N_VERSIONS}.0 with DVC")
    make_versions()
    print("\n  Git history (each line = one data version):")
    print("  " + git("log", "--oneline", "--decorate", "--no-color", quiet=True)
          .replace("\n", "\n  "))

    step(3, "start the MLflow tracking server")
    import mlflow_server
    proc = mlflow_server.start()

    step(4, "train on every data version, log to MLflow")
    train_all(args.retrain)

    step(5, "build report.html from MLflow")
    import report
    path = report.build_report()
    print(f"  report written to {path}")

    step(6, "double checkout: back to v1.0 and forward again")
    checkout_demo()

    print(f"\nAll done.\n  MLflow UI : {config.TRACKING_URI}\n"
          f"  Report    : {config.ROOT / 'report.html'}\n"
          f"  GUI       : streamlit run app.py")
    if args.stop_server and proc is not None:
        proc.terminate()
        print("  (MLflow server stopped)")


if __name__ == "__main__":
    main()
