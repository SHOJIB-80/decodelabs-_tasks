"""CLI entry point for IrisVision AI model comparison and sample prediction."""
from __future__ import annotations

import argparse

from irisvision.data import FEATURE_NAMES, load_dataset
from irisvision.evaluation import evaluate_models
from irisvision.models import MODEL_ORDER
from irisvision.prediction import predict_flower


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate four Iris classifiers and optionally predict a flower.")
    parser.add_argument("--predict", nargs=4, type=float, metavar=("SEPAL_LENGTH", "SEPAL_WIDTH", "PETAL_LENGTH", "PETAL_WIDTH"), help="Predict from four measurements in centimeters.")
    parser.add_argument("--model", choices=MODEL_ORDER, default=MODEL_ORDER[0], help="Model to use with --predict (default: original KNN baseline).")
    args = parser.parse_args()

    data = load_dataset()
    result = evaluate_models(data)
    print("IrisVision AI — stratified 80/20 holdout; 5-fold stratified CV for training-only model selection")
    print(result.metrics[["Model", "CV macro F1", "Test accuracy", "Test precision (macro)", "Test recall (macro)", "Test F1 (macro)"]].to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    if args.predict:
        outcome = predict_flower(result.models[args.model], args.predict)
        print(f"\nPrediction model: {args.model}")
        print("Measurements (cm): " + ", ".join(f"{name}={value:g}" for name, value in zip(FEATURE_NAMES, args.predict)))
        print(f"Predicted species: {outcome['species']}")
        if outcome["probabilities"]:
            print("Model probability estimates: " + ", ".join(f"{label} {value:.1%}" for label, value in outcome["probabilities"].items()))


if __name__ == "__main__":
    main()
