# Evaluation outputs

- `model_comparison.csv` contains the actual metrics and runtime measurements from a run of the delivered evaluation code against `data/iris.csv`.
- `classification_reports.json` contains the per-class precision, recall, F1 and support values for each fitted model on that run's final holdout.

These results use the fixed 80/20 stratified split (`random_state=42`), training-only 5-fold model selection, and macro-F1 scoring. Scores describe this small held-out subset; they are not guarantees of generalization. Runtime values depend on the computer and Python environment and should not be compared across machines as controlled benchmarks. Re-run `python main.py` after dependency installation to produce a fresh comparison in the terminal; the dashboard calculates its own current result at startup.
