"""Model pipelines and compact, reproducible training-only tuning grids."""
from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

MODEL_ORDER = [
    "K-Nearest Neighbors (KNN)",
    "Decision Tree",
    "Random Forest",
    "Support Vector Machine (SVM)",
]


def build_models() -> dict[str, Pipeline]:
    """Create fresh pipelines; scaling is fitted only within training folds."""
    return {
        MODEL_ORDER[0]: Pipeline([
            ("preprocess", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5)),
        ]),
        MODEL_ORDER[1]: Pipeline([
            ("preprocess", "passthrough"),
            ("classifier", DecisionTreeClassifier(random_state=42)),
        ]),
        MODEL_ORDER[2]: Pipeline([
            ("preprocess", "passthrough"),
            ("classifier", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)),
        ]),
        MODEL_ORDER[3]: Pipeline([
            ("preprocess", StandardScaler()),
            ("classifier", SVC(probability=True, random_state=42)),
        ]),
    }


def parameter_grids() -> dict[str, dict[str, list[object]]]:
    """Small grids selected before final testing; all search is confined to training folds."""
    return {
        MODEL_ORDER[0]: {
            "classifier__n_neighbors": [3, 5, 7, 9],
            "classifier__weights": ["uniform", "distance"],
            "classifier__p": [1, 2],
        },
        MODEL_ORDER[1]: {
            "classifier__criterion": ["gini", "entropy"],
            "classifier__max_depth": [None, 3, 5],
            "classifier__min_samples_leaf": [1, 2],
        },
        MODEL_ORDER[2]: {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 4],
            "classifier__min_samples_leaf": [1, 2],
        },
        MODEL_ORDER[3]: {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__kernel": ["rbf", "linear"],
            "classifier__gamma": ["scale"],
        },
    }
