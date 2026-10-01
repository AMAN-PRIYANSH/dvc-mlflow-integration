"""
data_versions.py - small helpers that ask Git and DVC about the data versions.

Who does what (the coat-check analogy):
    Git  keeps the small receipts  -> data/train.csv.dvc   (holds the md5 hash)
    DVC  keeps the real coats      -> the actual train.csv rows, in .dvc/cache

So:
    * to read the RECEIPT of a version we ask Git   (git show v2.0:data/train.csv.dvc)
    * to read the DATA of a version we ask DVC      (dvc.api.open(..., rev="v2.0"))

A "version" is simply a Git tag: v1.0, v2.0, v3.0, v4.0.
"""
import configparser
import subprocess
import sys

import dvc.api
import pandas as pd
import yaml

import config


def _git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=config.ROOT, capture_output=True,
                         text=True, check=True)
    return out.stdout.strip()


def ensure_local_remote() -> bool:
    """Point DVC at dvc_storage/ in .dvc/config.local, once per clone.

    .dvc/config says url = ../dvc_storage. For an old Git tag DVC reads that path
    from inside the tag, where it points to the drive root (C:\\dvc_storage), so
    `dvc fetch --all-tags` finds nothing. The local setting lives in the real folder,
    so it works for every tag. Returns True if it had to set it.
    """
    dvc_dir = config.ROOT / ".dvc"
    if not dvc_dir.is_dir():
        return False
    local = configparser.ConfigParser()
    local.read(dvc_dir / "config.local", encoding="utf-8")
    if any('remote "storage"' in s and local.has_option(s, "url") for s in local.sections()):
        return False
    subprocess.run([sys.executable, "-m", "dvc", "remote", "modify", "--local", "storage",
                    "url", config.DVC_REMOTE_DIR], cwd=config.ROOT, capture_output=True,
                   text=True, check=True)
    return True


def list_versions() -> list[str]:
    """All data versions = all Git tags that start with 'v', oldest first."""
    try:
        tags = _git("tag", "--list", "v*", "--sort=v:refname")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    return [t for t in tags.splitlines() if t]


def commit_of(rev: str) -> str:
    """The Git commit a tag points to (short form)."""
    return _git("rev-list", "-n", "1", "--abbrev-commit", rev)


def read_receipt(rev: str, path: str) -> dict:
    """Read the .dvc receipt file of `path` as it was at version `rev` (via Git)."""
    text = _git("show", f"{rev}:{path}.dvc")
    out = yaml.safe_load(text)["outs"][0]
    return {"md5": out["md5"], "size_bytes": out.get("size"), "text": text}


def read_data(rev: str, path: str) -> pd.DataFrame:
    """Ask DVC for the real data file exactly as it was at version `rev`."""
    with dvc.api.open(path, repo=str(config.ROOT), rev=rev) as f:
        return pd.read_csv(f)


def load_version(rev: str) -> dict:
    """Everything about one data version, in one dictionary."""
    train = read_data(rev, config.TRAIN_CSV)
    test = read_data(rev, config.TEST_CSV)
    train_receipt = read_receipt(rev, config.TRAIN_CSV)
    test_receipt = read_receipt(rev, config.TEST_CSV)

    versions = list_versions()
    idx = versions.index(rev) if rev in versions else -1
    if idx > 0:   # which rows are new compared with the previous version?
        prev_ids = set(read_data(versions[idx - 1], config.TRAIN_CSV)[config.ID_COL])
        new_rows = train[~train[config.ID_COL].isin(prev_ids)]
    else:
        new_rows = train

    return {
        "version": rev,
        "commit": commit_of(rev),
        "train": train,
        "test": test,
        "new_rows": new_rows,
        "train_md5": train_receipt["md5"],
        "test_md5": test_receipt["md5"],
        "train_receipt_text": train_receipt["text"],
        "test_receipt_text": test_receipt["text"],
    }


if __name__ == "__main__":
    for v in list_versions():
        info = load_version(v)
        print(f"{v}: commit {info['commit']}  train rows={len(info['train'])}  "
              f"new rows={len(info['new_rows'])}  md5={info['train_md5']}")
