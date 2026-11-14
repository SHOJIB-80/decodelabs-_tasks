# DecodeLabs AI Project 4 — VisionText AI (OCR)

VisionText AI is a small printed-text OCR application using Python, OpenCV, `pytesseract`, and the separately installed Tesseract OCR engine. It offers a command-line interface and a Streamlit dashboard.

## Features

- Accepts PNG, JPG, and JPEG files.
- Handles BGR color, grayscale, and transparent BGRA images; transparent pixels are composited onto white.
- Checks upload size and image dimensions before full decoding.
- Preserves the original grayscale, Gaussian blur, and adaptive-threshold variants and original selection rule.
- Filters recognized words by Tesseract's reported confidence and annotates accepted word boxes.
- Saves extracted text, a JSON report, the preprocessed image, and an annotated image from the CLI.
- Includes unit and actual-Tesseract integration tests plus a repeatable CER/WER benchmark.

Tesseract confidence is a model estimate, **not independently measured recognition accuracy**. Use ground-truth text and the benchmark to measure recognition errors.

## Requirements

- Python 3.10 or newer (tested here with Python 3.12).
- Tesseract OCR installed separately from Python packages, including English trained data (`eng`).
- Dependencies in `requirements.txt`.

### Install Tesseract on Windows

1. Install Tesseract using the [official installation documentation](https://tesseract-ocr.github.io/tessdoc/Installation.html). The commonly used Windows installer is linked from the [UB Mannheim Tesseract page](https://github.com/UB-Mannheim/tesseract/wiki).
2. Confirm that `tesseract.exe` is installed. The default expected path is `C:\Program Files\Tesseract-OCR\tesseract.exe`.
3. Open a new PowerShell terminal and check:

   ```powershell
   tesseract --version
   ```

4. If Tesseract is not on `PATH`, set the executable path for the current PowerShell session (adjust if installed elsewhere):

   ```powershell
   $env:TESSERACT_EXE = 'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```

### Set up in VS Code (PowerShell)

Extract the ZIP, open the extracted `DecodeLabs-AI-Project-4-OCR` folder in VS Code, then run these commands in its integrated PowerShell terminal:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If Tesseract was not added to `PATH`, set `$env:TESSERACT_EXE` in that same terminal before running commands. No manual file moves or import-path changes are needed.

## Run the project

Run OCR on the included sample and write results under `outputs/`:

```powershell
.\.venv\Scripts\python.exe main.py samples\sample_document.png
```

Optional CLI settings:

```powershell
.\.venv\Scripts\python.exe main.py samples\sample_document.png --confidence 80 --psm 6
```

Start the dashboard:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The dashboard retains the confidence slider, page segmentation selector, original/preprocessed/annotated previews, and extracted-text download. It also displays input dimensions, file size, and processing time.

### Configure input limits

Defaults are a **15 MB** upload limit and **25,000,000 decoded pixels**. Set environment variables before starting the CLI or dashboard to change them. Values must be positive integers.

```powershell
$env:VISIONTEXT_MAX_UPLOAD_MB = '20'
$env:VISIONTEXT_MAX_IMAGE_PIXELS = '30000000'
```

Large images that exceed the limits are rejected with an error. These checks protect both upload and local-file paths.

## Tests

Run the full suite:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The suite includes deterministic unit tests and integration tests that invoke the real Tesseract executable. Integration tests are skipped automatically if Tesseract is not installed. A separate Streamlit HTTP health check was also run during this project review (see verification summary in the delivery notes).

## Benchmark

Run the baseline-versus-upgraded benchmark from the project root:

```powershell
.\.venv\Scripts\python.exe benchmarks\run_benchmark.py --repeats 3
```

Each image requires an independently prepared UTF-8 reference text file at `benchmarks/ground_truth/<image-stem>.txt`. The included reference was manually transcribed from the sample image, not copied from OCR output. The script refuses to benchmark an image without a matching reference. It evaluates both existing preprocessing variants (grayscale and adaptive threshold), records CER, WER, average OCR-call time, accepted-word count, mean confidence, Tesseract version, and the sample-image SHA-256. Reports are written to `benchmarks/results/benchmark_results.csv` and `.json`.

CER/WER use Levenshtein edit distance after case-folding and whitespace tokenization; punctuation is retained. Timing measures the OCR call, not installation or file upload, and varies by machine and Tesseract build.

### Recorded result for the included sample

Environment: Tesseract 5.3.4, PSM 6, confidence threshold 80, three repeats. The one included sample produced exact normalized text against its separately stored reference in both preprocessing variants:

| Pipeline | Variant | CER | WER | Accepted words | Mean confidence | Mean OCR time |
|---|---|---:|---:|---:|---:|---:|
| Original baseline | Grayscale | 0.0000 | 0.0000 | 28 | 95.2% | 0.1803 s |
| Original baseline | Adaptive threshold | 0.0000 | 0.0000 | 28 | 95.1% | 0.1448 s |
| Upgraded | Grayscale | 0.0000 | 0.0000 | 28 | 95.2% | 0.1532 s |
| Upgraded | Adaptive threshold | 0.0000 | 0.0000 | 28 | 95.1% | 0.1545 s |

The OCR pipeline was intentionally not changed: no accuracy improvement is claimed. Small timing differences are within expected run-to-run/environment variation and are not evidence of a regression or speedup. The three-image baseline observations in the original upgrade brief could not be reproduced because those images were not present in the uploaded archive.

## Baseline and outputs

`benchmarks/baseline/` contains copies of the original application modules and original tests from the supplied archive for comparison. The active application keeps the same main function names, CLI options, preprocessing variants, result keys, confidence filtering, and dashboard features; it adds validation and a processing-time field.

The CLI creates:

- `outputs/extracted_text.txt` — accepted OCR text.
- `outputs/recognized_words.png` — annotated image with accepted word boxes.
- `outputs/preprocessed.png` — selected preprocessing result.
- `outputs/report.json` — text, word boxes, confidence values, and timing.

## Limitations

- The archive provides one clean, generated sample image, not a representative real-world benchmark set. Zero CER/WER on that sample does not predict performance on receipts, rotated photos, low-resolution scans, handwriting, or other languages.
- Deskewing, rotation correction, resizing, denoising, and contrast enhancement are not enabled. No enhancement was made default because one simple sample does not provide evidence of general improvement.
- OCR confidence is not calibrated accuracy; use a ground-truth evaluation for decisions that need verified correctness.
- Tesseract must be installed separately. The Python dependency `pytesseract` is only its interface.
- File-size and pixel limits can be configured through the environment variables described above.
