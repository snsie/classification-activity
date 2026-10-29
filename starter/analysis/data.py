"""Load the supplied datasets. You don't need to change this file.

    from analysis import data
    df = data.load("diabetes")              # every row, with its split
    X, y = data.rows("diabetes", "train")   # the training rows: the models learn from these
    X_test, y_test = data.rows("diabetes", "test")  # the test rows: used only to evaluate
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATASETS = ["readmission", "diabetes"]
ID_COLUMNS = ["row_id", "split"]
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
    df = pd.read_csv(DATA / f"{key}.csv", dtype={"split": str})
    for c in categorical(key):
        df[c] = df[c].astype(object)  # text categories; blank = missing
    return df


def load(key: str) -> pd.DataFrame:
    return _load(key).copy()


def rows(key: str, split: str) -> tuple[pd.DataFrame, list[int]]:
    """The predictors and labels of the supplied "train" or "test" rows."""
    df = _load(key)
    part = df[df.split == split]
    return part[predictors(key)].reset_index(drop=True), part[TARGET].astype(int).tolist()


def ids(key: str, split: str) -> list[int]:
    df = _load(key)
    return df.loc[df.split == split, "row_id"].astype(int).tolist()
