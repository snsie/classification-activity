"""Load the supplied datasets. You don't need to change this file.

    from analysis import data
    df = data.load("heart")              # every row, with split and fold
    X, y, folds = data.development("heart")
    X_test, y_test = data.test("heart")  # only `python -m analysis test` should call this
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATASETS = ["readmission", "diabetes", "heart"]
ID_COLUMNS = ["row_id", "split", "fold"]
TARGET = "target"


@lru_cache(maxsize=None)
def dictionary() -> dict:
    """data/datasets.json: every column's meaning, units, categories and "When is it known?"."""
    return json.loads((DATA / "datasets.json").read_text(encoding="utf-8"))


def predictors(key: str) -> list[str]:
    """The predictor columns: everything except the ID columns and the target."""
    return [c["name"] for c in dictionary()[key]["columns"] if c["kind"] != "id"]


def categorical(key: str) -> list[str]:
    return [c["name"] for c in dictionary()[key]["columns"] if c["kind"] == "category"]


@lru_cache(maxsize=None)
def _load(key: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"{key}.csv", dtype={"split": str, "fold": "Int64"})
    for c in categorical(key):
        df[c] = df[c].astype(object)  # text categories; blank = missing
    return df


def load(key: str) -> pd.DataFrame:
    return _load(key).copy()


def development(key: str) -> tuple[pd.DataFrame, list[int], list[int]]:
    """Development rows: predictors, labels and the supplied fold (1–5) of each row."""
    df = _load(key)
    dev = df[df.split == "dev"]
    return dev[predictors(key)].reset_index(drop=True), dev[TARGET].astype(int).tolist(), dev.fold.astype(int).tolist()


def development_ids(key: str) -> list[int]:
    df = _load(key)
    return df.loc[df.split == "dev", "row_id"].astype(int).tolist()


def test(key: str) -> tuple[pd.DataFrame, list[int]]:
    """Test rows. Locked: only `python -m analysis test` reads them, once."""
    df = _load(key)
    t = df[df.split == "test"]
    return t[predictors(key)].reset_index(drop=True), t[TARGET].astype(int).tolist()


def test_ids(key: str) -> list[int]:
    df = _load(key)
    return df.loc[df.split == "test", "row_id"].astype(int).tolist()
