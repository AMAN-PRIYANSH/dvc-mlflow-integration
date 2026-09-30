"""
prepare_data.py - writes the data files that DVC will version.

The dataset: Breast Cancer Wisconsin (569 patients, 30 measurements of the
cell nuclei in a scan). Goal: predict if a tumour is malignant (1) or benign (0).
It ships inside scikit-learn, so no internet is needed.

How the versions are made (the teacher's "4 lines, then 4 more lines" idea):

    data/test.csv   -> 20% of patients, put aside ONCE. Never changes.
                       Every version is tested on this same "exam paper",
                       so the scores of v1..v4 are fair to compare.

    data/train.csv  -> the other 80%, arriving in batches:
                       v1 = batch 1
                       v2 = batch 1 + batch 2
                       v3 = batch 1 + batch 2 + batch 3
                       v4 = all batches
                       Each version keeps the old rows and adds new rows below.

Usage:
    python prepare_data.py --version 1      (writes train.csv for v1 + test.csv)
    python prepare_data.py --version 3
"""
import argparse
import math

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

import config


def load_full_dataset() -> pd.DataFrame:
    """All 569 patients as one table, with readable column names."""
    raw = load_breast_cancer(as_frame=True).frame
    df = raw.rename(columns=lambda c: c.replace(" ", "_"))
    # scikit-learn stores 0 = malignant, 1 = benign.
    # We flip it so that 1 = malignant, because malignant is the thing we
    # are trying to catch (the "positive" class for precision / recall).
    df[config.TARGET] = 1 - df.pop("target")
    df.insert(0, config.ID_COL, range(1, len(df) + 1))
    return df


def split_pool_and_test(df: pd.DataFrame):
    """Put the fixed test set aside, and shuffle the rest into 'arrival order'."""
    pool, test = train_test_split(
        df,
        test_size=config.TEST_FRACTION,
        stratify=df[config.TARGET],       # same malignant/benign ratio in both parts
        random_state=config.SEED,
    )
    pool = pool.sample(frac=1.0, random_state=config.SEED)   # order in which batches "arrive"
    test = test.sort_values(config.ID_COL)
    return pool, test


def rows_in_version(version: int, pool_size: int) -> int:
    """How many training rows version k has: k batches of equal size."""
    batch = math.ceil(pool_size / config.N_VERSIONS)
    return min(version * batch, pool_size)


def write_version(version: int) -> None:
    if not 1 <= version <= config.N_VERSIONS:
        raise SystemExit(f"--version must be between 1 and {config.N_VERSIONS}")

    pool, test = split_pool_and_test(load_full_dataset())
    n = rows_in_version(version, len(pool))
    train = pool.iloc[:n]           # old rows first, new batch appended at the bottom

    config.DATA_DIR.mkdir(exist_ok=True)
    train.to_csv(config.ROOT / config.TRAIN_CSV, index=False)
    test.to_csv(config.ROOT / config.TEST_CSV, index=False)

    prev = rows_in_version(version - 1, len(pool)) if version > 1 else 0
    print(f"[data] v{version}.0 -> train.csv has {n} rows "
          f"({n - prev} new rows added), test.csv has {len(test)} rows (fixed)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", type=int, required=True, help="1, 2, 3 or 4")
    write_version(parser.parse_args().version)
