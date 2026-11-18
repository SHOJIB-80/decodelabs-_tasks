"""Prediction helpers shared by the dashboard, CLI, and tests."""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from .data import FEATURE_NAMES, validate_measurements


def predict_flower(model, values: Iterable[object]) -> dict[str, object]:
    """Predict a species and return model-produced class probability estimates."""
    row = validate_measurements(values)
    feature_frame = pd.DataFrame(row, columns=FEATURE_NAMES)
    prediction = str(model.predict(feature_frame)[0])
    probabilities = None
    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(feature_frame)[0]
        probabilities = {str(label): float(prob) for label, prob in zip(model.classes_, raw)}
    return {"species": prediction, "probabilities": probabilities}


def measurements_outside_training_range(values: Iterable[object], X: pd.DataFrame) -> list[str]:
    """Return a transparent extrapolation warning for measurements outside observed ranges."""
    row = validate_measurements(values)[0]
    return [
        FEATURE_NAMES[index]
        for index, value in enumerate(row)
        if value < float(X[FEATURE_NAMES[index]].min()) or value > float(X[FEATURE_NAMES[index]].max())
    ]
