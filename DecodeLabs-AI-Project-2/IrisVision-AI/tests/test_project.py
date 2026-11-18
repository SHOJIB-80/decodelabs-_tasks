from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from irisvision.data import FEATURE_NAMES, SPECIES_NAMES, load_dataset, validate_measurements
from irisvision.evaluation import RANDOM_STATE, TEST_SIZE, evaluate_models
from irisvision.models import MODEL_ORDER, build_models
from irisvision.prediction import predict_flower


@pytest.fixture(scope="session")
def dataset():
    return load_dataset()


@pytest.fixture(scope="session")
def evaluation(dataset):
    return evaluate_models(dataset)


def test_dataset_loading_dimensions_and_canonical_records(dataset):
    reference = load_iris(as_frame=True)
    np.testing.assert_allclose(dataset[FEATURE_NAMES].to_numpy(), reference.data[FEATURE_NAMES].to_numpy())
    assert dataset.shape == (150, 5)
    assert dataset["species"].tolist() == [reference.target_names[i] for i in reference.target]
    assert dataset["species"].value_counts().to_dict() == {"setosa": 50, "versicolor": 50, "virginica": 50}


def test_dataset_has_no_missing_values_and_preserves_source_duplicate_count(dataset):
    reference = load_iris(as_frame=True)
    assert not dataset.isna().any().any()
    assert int(dataset.duplicated().sum()) == int(pd.concat([reference.data, pd.Series([reference.target_names[i] for i in reference.target], name="species")], axis=1).duplicated().sum())
    assert int(dataset.duplicated().sum()) == 1  # present in the canonical source; intentionally retained


@pytest.mark.parametrize("values", [
    [5.1, 3.5, 1.4],
    [5.1, 3.5, 1.4, 0.2, 9],
    [5.1, "not-a-number", 1.4, 0.2],
    [5.1, np.nan, 1.4, 0.2],
    [5.1, np.inf, 1.4, 0.2],
    [5.1, 3.5, 1.4, 0],
    [-1, 3.5, 1.4, 0.2],
])
def test_measurement_validation_rejects_invalid_inputs(values):
    with pytest.raises(ValueError):
        validate_measurements(values)


def test_measurement_validation_accepts_four_positive_finite_values():
    result = validate_measurements([5.1, 3.5, 1.4, 0.2])
    assert result.shape == (1, 4)
    assert np.isfinite(result).all()


def test_all_four_pipelines_train_and_predict(dataset):
    X, y = dataset[FEATURE_NAMES], dataset["species"]
    models = build_models()
    assert list(models) == MODEL_ORDER
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    for model in models.values():
        model.fit(X_train, y_train)
        output = model.predict(X_test)
        assert len(output) == len(X_test)
        assert set(output).issubset(set(SPECIES_NAMES))


@pytest.mark.parametrize("model_name", MODEL_ORDER)
def test_trained_prediction_has_valid_probabilities(evaluation, model_name):
    output = predict_flower(evaluation.models[model_name], [5.1, 3.5, 1.4, 0.2])
    assert output["species"] in SPECIES_NAMES
    probabilities = output["probabilities"]
    assert set(probabilities) == set(SPECIES_NAMES)
    assert all(0.0 <= probability <= 1.0 for probability in probabilities.values())
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_evaluation_contains_all_models_and_computed_multiclass_metrics(evaluation):
    assert evaluation.metrics["Model"].tolist() == MODEL_ORDER
    assert len(evaluation.metrics) == 4
    for metric in ["CV macro F1", "Test accuracy", "Test precision (macro)", "Test recall (macro)", "Test F1 (macro)"]:
        assert evaluation.metrics[metric].between(0.0, 1.0).all()
    assert len(evaluation.X_test) == 30
    assert len(evaluation.X_train) == 120
    for model in MODEL_ORDER:
        matrix = evaluation.confusion_matrices[model]
        assert matrix.shape == (3, 3)
        assert matrix.sum() == len(evaluation.y_test)
        assert len(evaluation.predictions[model]) == len(evaluation.y_test)
        assert set(evaluation.reports[model]).issuperset(set(SPECIES_NAMES))


def test_scaling_is_fitted_on_training_partition_only(evaluation):
    train_mean = evaluation.X_train[FEATURE_NAMES].mean().to_numpy()
    full_mean = load_dataset()[FEATURE_NAMES].mean().to_numpy()
    assert not np.allclose(train_mean, full_mean)
    for model_name in [MODEL_ORDER[0], MODEL_ORDER[3]]:
        scaler = evaluation.models[model_name].named_steps["preprocess"]
        assert isinstance(scaler, StandardScaler)
        np.testing.assert_allclose(scaler.mean_, train_mean)
    for model_name in [MODEL_ORDER[1], MODEL_ORDER[2]]:
        assert evaluation.models[model_name].named_steps["preprocess"] == "passthrough"


def test_train_and_test_records_do_not_overlap(evaluation):
    assert set(evaluation.X_train.index).isdisjoint(set(evaluation.X_test.index))
    assert set(evaluation.y_train.index) == set(evaluation.X_train.index)
    assert set(evaluation.y_test.index) == set(evaluation.X_test.index)


def test_evaluation_is_reproducible(dataset, evaluation):
    second = evaluate_models(dataset)
    deterministic_columns = [
        "Model", "CV macro F1", "Test accuracy", "Test precision (macro)",
        "Test recall (macro)", "Test F1 (macro)", "Best parameters",
    ]
    pd.testing.assert_frame_equal(
        evaluation.metrics[deterministic_columns],
        second.metrics[deterministic_columns],
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    for name in MODEL_ORDER:
        np.testing.assert_array_equal(evaluation.predictions[name], second.predictions[name])
        np.testing.assert_array_equal(evaluation.confusion_matrices[name], second.confusion_matrices[name])
