"""
mlflow_server.py - starts the MLflow tracking server (if it is not running yet).

    Backend store  = sqlite:///mlflow.db  -> a database file that stores every run:
                                            parameters, metrics, tags, run names.
    Artifact store = ./mlartifacts        -> the files of every run:
                                            plots (ROC, confusion matrix), saved models.

train.py and the GUI never touch those files directly. They talk to the server
at http://127.0.0.1:5000, and the server writes into the database and folder.
That same address is also the MLflow web UI you open in the browser.

Run by hand:   python mlflow_server.py
"""
import subprocess
import sys
import time
import urllib.request

import config


def is_running(timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(f"{config.TRACKING_URI}/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def server_command() -> list[str]:
    return [
        sys.executable, "-m", "mlflow", "server",
        "--backend-store-uri", config.BACKEND_STORE_URI,
        "--artifacts-destination", config.ARTIFACTS_DESTINATION,
        "--host", config.MLFLOW_HOST,
        "--port", str(config.MLFLOW_PORT),
    ]


def start(wait_seconds: int = 90) -> subprocess.Popen | None:
    """Start the server in the background. Returns the process (None if already up)."""
    if is_running():
        print(f"[mlflow] server already running at {config.TRACKING_URI}")
        return None

    log = open(config.ROOT / "mlflow_server.log", "a", encoding="utf-8")
    kwargs = {}
    if sys.platform == "win32":
        # its own background process, so closing the GUI does not kill MLflow
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    proc = subprocess.Popen(server_command(), cwd=config.ROOT, stdout=log,
                            stderr=subprocess.STDOUT, **kwargs)

    print(f"[mlflow] starting server at {config.TRACKING_URI} ...", flush=True)
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if is_running():
            print("[mlflow] server is up")
            return proc
        if proc.poll() is not None:
            raise RuntimeError("MLflow server stopped. Look inside mlflow_server.log")
        time.sleep(1)
    raise RuntimeError("MLflow server did not start in time. Look inside mlflow_server.log")


if __name__ == "__main__":
    p = start()
    if p is not None:
        print("Press Ctrl+C to stop the MLflow server.")
        try:
            p.wait()
        except KeyboardInterrupt:
            p.terminate()
