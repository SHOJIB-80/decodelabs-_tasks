# IrisVision AI

A local-first Iris classification workbench that preserves the original KNN implementation and expands it into a tested, interactive Streamlit dashboard. It compares K-Nearest Neighbors (KNN), Decision Tree, Random Forest, and Support Vector Machine (SVM) using actual trained scikit-learn pipelines.

## Features

- Interactive Streamlit dashboard with overview, prediction, model comparison, dataset explorer, and model evaluation sections.
- Manual measurement entry or sliders for the four measurements in centimeters.
- Select any of four trained classifiers and view model-produced species probabilities.
- Training-only stratified cross-validation for model selection; one reserved stratified holdout for final comparison.
- Macro-averaged precision, recall and F1, accuracy, confusion matrices, classification reports, and measured runtime.
- Interactive Plotly charts for class counts, feature distributions and relationships, metric comparison, runtime, probabilities, and confusion matrices.
- Command-line model evaluation and prediction, plus automated tests.
- No paid APIs, external credentials, runtime internet access, or generated/fabricated example records.

## Dataset and integrity

`data/iris.csv` contains the 150 canonical Iris observations (four numeric features and the original species labels) exported from scikit-learn's bundled `load_iris` copy of the classic Fisher Iris dataset. The project ships the records locally and does not fetch them from the internet. All records are retained; the canonical data includes **one repeated complete row**, which is not removed or silently modified. The three classes are setosa, versicolor, and virginica, with 50 records each.

The legacy script loaded the same bundled dataset at runtime. Its original KNN flow remains available, byte-for-byte, at `legacy/main.py`; the original console capture is retained in `assets/original_console_run.png`.

## Architecture

```text
app.py                    Streamlit dashboard and navigation
main.py                   CLI model comparison and optional prediction
irisvision/data.py        Dataset loading and input validation
irisvision/models.py      Four pipelines and tuning grids
irisvision/evaluation.py  Stratified split, training-only CV and final metrics
irisvision/prediction.py  Validated model-backed predictions
irisvision/visualizations.py  Plotly chart builders
data/iris.csv             Canonical, local Iris records
data/README.md            Dataset source, schema, and integrity notes
legacy/main.py            Unmodified original KNN script
assets/original_console_run.png  Original output capture
tests/                    Dataset, model, evaluation, and input tests
results/                  Actual comparison CSV and classification reports
requirements.txt          Pinned runtime/test dependencies
```

## Algorithms

- **KNN:** `StandardScaler` followed by a 5-neighbor KNN baseline, with its parameters tuned only within training folds. This preserves the original model family and preprocessing approach.
- **Decision Tree:** deterministic `DecisionTreeClassifier`; scaling is not applied because tree split decisions do not require feature standardization.
- **Random Forest:** deterministic 200-tree baseline; its training-only search considers tree count and depth/leaf settings.
- **SVM:** `StandardScaler` followed by `SVC(probability=True)`, so the dashboard can display model-produced probability estimates.

All preprocessing is encapsulated in scikit-learn pipelines. Scalers are fit separately inside each cross-validation training fold, then refit on the complete training partition by the selected search. The holdout is not supplied to `GridSearchCV`.

## Evaluation methodology

The default, reproducible partition is a stratified 80/20 train/test split with `random_state=42`. The 120-record training partition is used for stratified 5-fold cross-validation. Each model's hyperparameters are selected by **macro F1 on training folds**; after selection, the fitted model is evaluated once on the 30-record holdout. The same split and metrics are used for all four models.

- **Validation/CV performance** is an average across training-only folds and informs parameter selection.
- **Final test performance** is measured on records withheld from that selection procedure. It is a more independent estimate for this split, not a guarantee of performance on future, different, or larger populations.
- Precision, recall and F1 use macro averaging, giving each of the three species equal weight. The classification report also shows per-species results and support.
- Training time includes cross-validation search and final refit; prediction timing is a small local measurement and is hardware/runtime dependent. Neither is a stable benchmark across computers.

No algorithm is required to achieve a predetermined accuracy. Scores will vary if the data, split, library versions, or evaluation design change.

The packaged run is recorded in `results/model_comparison.csv` and `results/classification_reports.json`. On that run, KNN and Decision Tree each scored 93.33% test accuracy, Random Forest 96.67%, and SVM 93.33%; the best 5-fold training CV macro F1 was 0.975 for SVM. These figures describe this one small holdout only and do not establish that one model generalizes best. Runtime measures are machine-dependent.

## Windows setup (PowerShell / Visual Studio Code)

Use Python 3.11 or 3.12. Extract the ZIP first, then open the extracted folder in Visual Studio Code. In PowerShell, change to that folder (adjust the path as needed):

```powershell
cd "$HOME\Downloads\IrisVision-AI"
py -3.11 -m venv .venv
# If Python 3.11 is not installed, use this instead:
# py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, either allow activation for the current terminal with `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and activate again, or call the environment's interpreter directly as shown below. After dependencies have been installed, the dashboard and tests run offline.

### Run the dashboard

```powershell
streamlit run app.py
```

Streamlit prints a local address; open it in your browser. Use the left navigation to switch between the five sections. Select the active classifier in the sidebar.

### Run the CLI

```powershell
python main.py
python main.py --predict 5.1 3.5 1.4 0.2
python main.py --predict 6.0 2.9 4.5 1.5 --model "Support Vector Machine (SVM)"
```

The first command evaluates the four models. `--predict` accepts sepal length, sepal width, petal length, and petal width, in that order and in centimeters. The default prediction model is the original KNN baseline. Predictions run through a fitted pipeline; no output label is hard-coded.

### Run tests

```powershell
python -m pytest -q
```

In VS Code, choose `.venv\Scripts\python.exe` as the Python interpreter if it is not selected automatically.

## Screenshots

The original project console capture is included at [`assets/original_console_run.png`](assets/original_console_run.png). The new dashboard renders its dataset and evaluation visualizations interactively when run locally.

## Known limitations

- Iris is a tiny, balanced, well-studied teaching dataset; its scores are not evidence of real-world or clinical performance.
- A predicted class and its probabilities are model outputs, not a causal explanation or certainty. SVM probability estimates are produced by scikit-learn's internal probability estimation.
- Measurements outside the range observed in the training partition are allowed with an extrapolation warning; classification beyond the training data may be unreliable.
- Timing numbers are local diagnostics, not controlled performance benchmarks.
- The dashboard trains and tunes its small models locally when first loaded; no pre-trained model binary is bundled.
- PowerShell installation commands are documented for Windows; verification in this delivery was run in the available Linux sandbox, not on a Windows device.
