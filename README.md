# decodelabs-_tasks

A collection of four local Python applications developed for DecodeLabs AI tasks: an offline intent-based chatbot, an Iris classification workbench, a career-exploration recommender, and a printed-text OCR application.

The projects use Streamlit dashboards, command-line interfaces, and local datasets or knowledge bases. They do not share a common server or database.

## Projects

| Project | Purpose | Primary interfaces |
|---|---|---|
| [DecodeBot AI](DecodeLabs-AI-Project-1/DecodeBot-AI/) | Answers educational questions through local TF-IDF, character similarity, fuzzy matching, and curated intent responses | Streamlit dashboard, CLI |
| [IrisVision AI](DecodeLabs-AI-Project-2/IrisVision-AI/) | Compares four classifiers on the canonical Iris dataset and produces model-backed flower predictions | Streamlit dashboard, CLI |
| [SkillPath AI](SkillPath-AI-DecodeLabs-Project-3/SkillPath-AI/) | Recommends technology careers from user skills using processed O*NET 31.0 occupation profiles | Streamlit dashboard, CLI |
| [VisionText AI](DecodeLabs-AI-Project-4-OCR/DecodeLabs-AI-Project-4-OCR/) | Extracts printed text from PNG and JPEG images and produces annotated OCR outputs | Streamlit dashboard, CLI |

## Repository Structure

```text
.
├── DecodeLabs-AI-Project-1/
│   └── DecodeBot-AI/
│       ├── app.py                  # Streamlit chatbot
│       ├── chatbot.py              # CLI chatbot
│       ├── conversation.py         # Session history and follow-ups
│       ├── nlp_engine.py           # Intent matching engine
│       ├── knowledge_base/         # Local intents, examples, and responses
│       └── tests/
├── DecodeLabs-AI-Project-2/
│   └── IrisVision-AI/
│       ├── app.py                  # Streamlit ML workbench
│       ├── main.py                 # Model evaluation and prediction CLI
│       ├── irisvision/             # Data, models, evaluation, prediction, charts
│       ├── data/iris.csv           # Bundled Iris records
│       └── tests/
├── SkillPath-AI-DecodeLabs-Project-3/
│   └── SkillPath-AI/
│       ├── app.py                  # Streamlit recommender
│       ├── main.py                 # CLI recommender
│       ├── import_onet.py          # Rebuild processed O*NET profiles
│       ├── software_skills.csv     # Original O*NET 31.0 source
│       ├── raw_skills.csv          # Processed occupation profiles
│       └── skillpath/              # Data, normalization, and ranking modules
└── DecodeLabs-AI-Project-4-OCR/
    └── DecodeLabs-AI-Project-4-OCR/
        ├── app.py                  # Streamlit OCR dashboard
        ├── main.py                 # CLI OCR runner
        ├── ocr_engine.py           # Image handling and OCR pipeline
        ├── benchmarks/             # Baseline and benchmark assets
        ├── samples/                # Bundled sample image
        └── tests/
```

## Technology Summary

| Area | Technologies |
|---|---|
| Applications | Python 3, Streamlit |
| Text matching | scikit-learn TF-IDF and cosine similarity, Python `difflib` |
| Machine learning | scikit-learn KNN, decision tree, random forest, and SVM pipelines |
| Career data | Pandas, NumPy, scikit-learn, processed O*NET 31.0 Software Skills data |
| OCR | OpenCV, Tesseract OCR, `pytesseract`, Pillow, NumPy |
| Visualization | Streamlit, Plotly in IrisVision AI |
| Testing | `unittest`, `pytest` |

Python 3.10 or newer is supported by the project documentation. Individual dependencies and exact versions are declared in each project directory's `requirements.txt`.

## Setup

Each application is independent. Open one project folder in a terminal or editor and create a virtual environment inside that folder.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The same setup works in PowerShell when run from the selected project directory. Installation requires an internet connection; the applications use local resources after dependencies and, for OCR, the separate Tesseract executable are available.

## 1. DecodeBot AI

DecodeBot AI is an offline intent-based educational chatbot for topics including Python, programming, artificial intelligence, and machine learning. It normalizes text, compares it against local intent examples with TF-IDF and character features, combines cosine and fuzzy scores, and applies curated cue scores. The configured confidence threshold is `0.48`.

```powershell
cd DecodeLabs-AI-Project-1\DecodeBot-AI
python -m pip install -r requirements.txt
python chatbot.py
python -m streamlit run app.py
python -m unittest discover -s tests -v
python evaluate.py
```

The Streamlit interface maintains session-scoped conversation history and clears it with the sidebar control. The CLI supports `exit`, `quit`, and `/clear`. The knowledge base is [`knowledge_base/intents.json`](DecodeLabs-AI-Project-1/DecodeBot-AI/knowledge_base/intents.json).

DecodeBot is a text-matching chatbot, not a generative language model or LLM. It has no API-key or external-service integration.

## 2. IrisVision AI

IrisVision AI provides a reproducible, local classifier workbench for the classic Iris dataset. It contains 150 records with sepal and petal measurements and three species labels.

The implemented models are:

- K-Nearest Neighbors with standardization
- Decision Tree
- Random Forest
- Support Vector Machine with standardization and probability estimates

The evaluation uses a stratified 80/20 split with `random_state=42`, with 5-fold stratified cross-validation for training-only model selection. The remaining 30 records form the final holdout used for reported test metrics.

```powershell
cd DecodeLabs-AI-Project-2\IrisVision-AI
python -m pip install -r requirements.txt
python -m streamlit run app.py
python main.py
python main.py --predict 5.1 3.5 1.4 0.2
python main.py --predict 6.0 2.9 4.5 1.5 --model "Support Vector Machine (SVM)"
python -m pytest -q
```

The interactive dashboard includes dataset exploration, four-model comparison, live prediction, and holdout evaluation sections. The canonical source dataset and test data are bundled locally; no external API or database is used.

## 3. SkillPath AI

SkillPath AI ranks occupations by overlap with at least three selected or custom technology skills. The project includes the original O*NET 31.0 Software Skills CSV, processes it into occupation profiles, and applies a combined score of 65% TF-IDF/cosine similarity and 35% inverse-frequency-weighted skill coverage. Broad software tags have a configured scoring penalty.

```powershell
cd SkillPath-AI-DecodeLabs-Project-3\SkillPath-AI
python -m pip install -r requirements.txt
python import_onet.py
python -m streamlit run app.py
python main.py
python -m unittest discover -s tests -v
```

`software_skills.csv` is the source file. `import_onet.py` regenerates `raw_skills.csv` and `data_manifest.json`; the manifest records source SHA-256, row counts, duplicate and incomplete-row handling, profile count, and source version. The CLI uses the configured technology occupation scope, while the dashboard also exposes all-occupation and other configured scopes.

The combined score describes profile overlap. It is not a probability of hiring, career eligibility, or career success. O*NET workplace software examples are not a complete inventory of job requirements.

O*NET 31.0 Database data is used under CC BY 4.0. Attribution and source notices are retained in [`NOTICE.md`](SkillPath-AI-DecodeLabs-Project-3/SkillPath-AI/NOTICE.md).

## 4. VisionText AI

VisionText AI processes PNG, JPG, and JPEG images with OpenCV and Tesseract OCR. It tests original grayscale and adaptive-threshold preprocessing variants, recognizes word boxes, filters them by Tesseract confidence, and retains the result with the greater accepted-word count and confidence average.

Tesseract must be installed separately:

- Add `tesseract` to `PATH`, or
- Set `TESSERACT_EXE` to the Tesseract executable path before starting the application.

```powershell
$env:TESSERACT_EXE = 'C:\Program Files\Tesseract-OCR\tesseract.exe'
cd DecodeLabs-AI-Project-4-OCR\DecodeLabs-AI-Project-4-OCR
python -m pip install -r requirements.txt
python main.py samples\sample_document.png
python main.py samples\sample_document.png --confidence 80 --psm 6
python -m streamlit run app.py
python -m unittest discover -s tests -v
python benchmarks\run_benchmark.py --repeats 3
```

The CLI writes:

- `outputs/extracted_text.txt`
- `outputs/report.json`
- `outputs/preprocessed.png`
- `outputs/recognized_words.png`

OCR input is limited to 15 MB and 25,000,000 decoded pixels by default. They can be changed with positive-integer environment variables:

```powershell
$env:VISIONTEXT_MAX_UPLOAD_MB = '20'
$env:VISIONTEXT_MAX_IMAGE_PIXELS = '30000000'
```

Tesseract confidence is an OCR-engine estimate, not independently measured recognition accuracy. CER and WER benchmarks require prepared ground-truth text and are not claims of general real-world accuracy.

## Running Tests

The test systems differ by project:

| Project | Command |
|---|---|
| DecodeBot AI | `python -m unittest discover -s tests -v` |
| IrisVision AI | `python -m pytest -q` |
| SkillPath AI | `python -m unittest discover -s tests -v` |
| VisionText AI | `python -m unittest discover -s tests -v` |

VisionText AI integration tests invoke Tesseract and skip automatically if the executable is unavailable. No repository-wide test runner is configured.

## Included Screenshots and Outputs

The repository contains project screenshots and generated OCR/benchmark outputs, including:

- `DecodeLabs-AI-Project-1/DecodeBot-AI/assets/decodebot_original_demo.png`
- `DecodeLabs-AI-Project-2/IrisVision-AI/assets/original_console_run.png`
- `SkillPath-AI-DecodeLabs-Project-3/SkillPath-AI/` dashboard captures
- `DecodeLabs-AI-Project-4-OCR/DecodeLabs-AI-Project-4-OCR/VisionText AI OCR Dashboard-1.png`
- `DecodeLabs-AI-Project-4-OCR/DecodeLabs-AI-Project-4-OCR/VisionText AI OCR Dashboard-2.png`

## Known Limitations

- The projects are independent local applications; one virtual environment or one set of commands does not start all of them.
- DecodeBot AI can only answer the concepts represented by its local knowledge base and is not a general-purpose language model.
- IrisVision AI uses a small educational dataset and a fixed stratified holdout; its scores do not establish medical or real-world performance.
- SkillPath AI is a career-exploration tool based on workplace software mentions, not employment eligibility or verified job requirements.
- VisionText AI requires a separately installed Tesseract executable and its recorded benchmark covers only the included sample, not a general OCR dataset.
- No database, backend server, external credential configuration, deployment configuration, container setup, or CI workflow is present at the repository root.
- A license file is not present in the repository.