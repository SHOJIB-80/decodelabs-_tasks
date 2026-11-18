"""Reproducible model selection and one-time final holdout evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .data import FEATURE_NAMES, SPECIES_NAMES, load_dataset
from .models import MODEL_ORDER, build_models, parameter_grids

RANDOM_STATE = 42
TEST_SIZE = 0.20


@dataclass
class EvaluationResult:
    """Models, split data, predictions, and comparable final metrics."""
    models: dict[str, object]
    metrics: pd.DataFrame
    reports: dict[str, dict]
    confusion_matrices: dict[str, np.ndarray]
    predictions: dict[str, np.ndarray]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def evaluate_models(dataset: pd.DataFrame | None = None) -> EvaluationResult:
    """Tune with stratified CV on training data; use the untouched holdout once."""
    frame = load_dataset() if dataset is None else dataset.copy()
    X = frame[FEATURE_NAMES]
    y = frame["species"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    base_models = build_models()
    grids = parameter_grids()
    fitted: dict[str, object] = {}
    predictions: dict[str, np.ndarray] = {}
    reports: dict[str, dict] = {}
    matrices: dict[str, np.ndarray] = {}
    rows: list[dict[str, float | str]] = []

    for name in MODEL_ORDER:
        search = GridSearchCV(
            estimator=base_models[name],
            param_grid=grids[name],
            scoring="f1_macro",
            cv=cv,
            refit=True,
            n_jobs=1,
            return_train_score=False,
        )
        started = perf_counter()
        search.fit(X_train, y_train)
        training_seconds = perf_counter() - started
        model = search.best_estimator_
        fitted[name] = model

        # The final test split is accessed only after model selection is complete.
        started = perf_counter()
        prediction = model.predict(X_test)
        prediction_seconds = perf_counter() - started
        predictions[name] = prediction
        reports[name] = classification_report(
            y_test,
            prediction,
            labels=SPECIES_NAMES,
            target_names=SPECIES_NAMES,
            output_dict=True,
            zero_division=0,
        )
        matrices[name] = confusion_matrix(y_test, prediction, labels=SPECIES_NAMES)
        rows.append({
            "Model": name,
            "CV macro F1": float(search.best_score_),
            "Test accuracy": float(accuracy_score(y_test, prediction)),
            "Test precision (macro)": float(precision_score(y_test, prediction, labels=SPECIES_NAMES, average="macro", zero_division=0)),
            "Test recall (macro)": float(recall_score(y_test, prediction, labels=SPECIES_NAMES, average="macro", zero_division=0)),
            "Test F1 (macro)": float(f1_score(y_test, prediction, labels=SPECIES_NAMES, average="macro", zero_division=0)),
            "Training time (s)": float(training_seconds),
            "Prediction time (ms/sample)": float(prediction_seconds * 1000 / len(X_test)),
            "Best parameters": str(search.best_params_),
        })

    return EvaluationResult(
        models=fitted,
        metrics=pd.DataFrame(rows),
        reports=reports,
        confusion_matrices=matrices,
        predictions=predictions,
        X_train=X_train.copy(),
        X_test=X_test.copy(),
        y_train=y_train.copy(),
        y_test=y_test.copy(),
    )
