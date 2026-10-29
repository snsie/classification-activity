"""The reference models, built to the spec in starter/AGENT_BRIEF.md (Models).

The starter ships no analysis/models.py: each student's AI agent writes it. build_results.py uses these
for the baseline-only starter results and for the reference run whose test AUROCs check.py compares
every student's results with.
"""

from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis.data import encode  # the starter's own first step (build_results puts starter/ on sys.path)


def baseline():
    return DummyClassifier(strategy="prior")


def logreg():
    return make_pipeline(encode(), SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=5000))


def knn():
    return make_pipeline(encode(), SimpleImputer(strategy="median"), StandardScaler(), KNeighborsClassifier(n_neighbors=25))


def rf():
    return make_pipeline(encode(), SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=-1))


def lgbm():
    return make_pipeline(encode(), LGBMClassifier(random_state=0, verbose=-1, n_jobs=8))


MODELS = {"baseline": baseline, "logreg": logreg, "knn": knn, "rf": rf, "lgbm": lgbm}
