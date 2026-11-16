"""Evaluate intent matching against held-out examples not used for training."""
from __future__ import annotations

import json
from pathlib import Path

from nlp_engine import DEFAULT_KNOWLEDGE_PATH, IntentEngine, KnowledgeBase

ROOT = Path(__file__).parent
EVALUATION_PATH = ROOT / "tests" / "evaluation_set.json"


def calculate_metrics(expected: list[str], predicted: list[str], labels: list[str]) -> dict[str, float]:
    if not expected or len(expected) != len(predicted):
        raise ValueError("Expected and predicted labels must be non-empty lists of equal length.")
    accuracy = sum(gold == guess for gold, guess in zip(expected, predicted)) / len(expected)
    precisions, recalls, f1s = [], [], []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(expected, predicted))
        fp = sum(g != label and p == label for g, p in zip(expected, predicted))
        fn = sum(g == label and p != label for g, p in zip(expected, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return {"accuracy": accuracy, "macro_precision": sum(precisions)/len(labels),
            "macro_recall": sum(recalls)/len(labels), "macro_f1": sum(f1s)/len(labels)}


def main() -> int:
    with EVALUATION_PATH.open(encoding="utf-8") as handle:
        cases = json.load(handle)
    engine = IntentEngine(KnowledgeBase.load(DEFAULT_KNOWLEDGE_PATH))
    expected = [case["intent"] for case in cases]
    matches = [engine.match(case["text"]) for case in cases]
    predicted = [match.intent if match else "unknown" for match in matches]
    labels = [*engine.knowledge.intents, "unknown"]
    metrics = calculate_metrics(expected, predicted, labels)
    known_count = sum(label != "unknown" for label in expected)
    unknown_count = len(expected) - known_count
    print(f"Held-out evaluation examples: {len(cases)} ({known_count} in-scope; {unknown_count} out-of-scope)")
    print("Prompts are separate from the knowledge-base training examples.")
    for name, value in metrics.items():
        print(f"{name.replace('_', ' ').title():<18}: {value:.3f}")
    errors = [(case["text"], case["intent"], pred)
              for case, pred in zip(cases, predicted) if case["intent"] != pred]
    print(f"Incorrect predictions: {len(errors)}")
    for text, gold, pred in errors:
        print(f"  {text!r}: expected {gold}, got {pred}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
