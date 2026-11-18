"""Canonical Iris dataset loading and feature validation utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "iris.csv"
FEATURE_NAMES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]
SPECIES_NAMES = ["setosa", "versicolor", "virginica"]


def load_dataset(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the preserved, real 150-record Iris dataset shipped with the project."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Iris data file was not found: {dataset_path}")
    frame = pd.read_csv(dataset_path)
    expected = [*FEATURE_NAMES, "species"]
    if list(frame.columns) != expected:
        raise ValueError(f"Dataset columns must be exactly: {expected}")
    if frame.empty:
        raise ValueError("The Iris dataset is empty.")
    if frame[expected].isna().any().any():
        raise ValueError("The Iris dataset contains missing values.")
    if set(frame["species"].unique()) != set(SPECIES_NAMES):
        raise ValueError("The dataset must contain the three canonical Iris species.")
    return frame


def validate_measurements(values: Iterable[object]) -> np.ndarray:
    """Return four finite, positive measurements, or raise a useful ValueError."""
    try:
        row = np.asarray(list(values), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Enter four numeric measurements in centimeters.") from exc
    if row.shape != (4,):
        raise ValueError("Exactly four measurements are required, in sepal-length, sepal-width, petal-length, petal-width order.")
    if not np.isfinite(row).all():
        raise ValueError("Measurements must be finite numbers; NaN and infinity are not valid.")
    if (row <= 0).any():
        raise ValueError("All four flower measurements must be greater than zero.")
    return row.reshape(1, -1)
