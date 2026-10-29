"""The five models. Everyone compares the same models with the same settings.

Every model is built fresh by a function, as one scikit-learn Pipeline. The whole pipeline is
fitted on the training rows only, so filling in missing values and scaling learn from training
rows only (lecture §14, rule 2).

YOUR JOB: complete the TODO below, `knn()`, following the `logreg()` example.
Don't change the other models or any settings: `python check.py` compares them with the reference.
"""

from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier  # noqa: F401  (for your kNN pipeline)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 0


def encode():
    """Supplied first step: turns each text category (such as admission type) into 0/1 columns, learned from the training rows only."""
    return make_column_transformer(
        (OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10, sparse_output=False),
         make_column_selector(dtype_include=object)),
        remainder="passthrough",
    )


def baseline():
    """Prevalence baseline: every patient gets the share of positives in the training rows."""
    return DummyClassifier(strategy="prior")


def logreg():
    """Logistic regression. WORKED EXAMPLE: copy this shape for knn()."""
    return make_pipeline(
        encode(),                              # 1. text categories → 0/1 columns
        SimpleImputer(strategy="median"),      # 2. fill missing values with the training median
        StandardScaler(),                      # 3. put every feature on the same scale
        LogisticRegression(max_iter=5000),     # 4. the classifier
    )


def knn():
    """k-nearest neighbours with k = 25.

    TODO: return a pipeline like logreg(): encode() → median imputation → scaling →
    KNeighborsClassifier(n_neighbors=25). Scaling must be inside the pipeline.
    """
    raise NotImplementedError("TODO: complete knn() in analysis/models.py, following the logreg() example.")


def rf():
    """Random forest: 300 trees, at least 3 training patients per leaf (as in the lecture)."""
    return make_pipeline(
        encode(),
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1),
    )


def lgbm():
    """Gradient boosting with LightGBM's defaults, as in the lecture (it handles missing values itself)."""
    return make_pipeline(encode(), LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=8))


# Name → builder. The order is the order results are written and shown.
MODELS = {"baseline": baseline, "logreg": logreg, "knn": knn, "rf": rf, "lgbm": lgbm}
