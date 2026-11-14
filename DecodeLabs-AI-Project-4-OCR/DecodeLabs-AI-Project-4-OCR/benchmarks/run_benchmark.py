"""Compare the preserved baseline and current OCR against independent ground truth."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import pytesseract
from pytesseract import Output

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ocr_engine as upgraded  # noqa: E402


def load_baseline_module():
    path = ROOT / "benchmarks" / "baseline" / "ocr_engine.py"
    spec = importlib.util.spec_from_file_location("visiontext_baseline_ocr_engine", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load preserved baseline: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (left_item != right_item),
            ))
        previous = current
    return previous[-1]


def normalized_tokens(text: str) -> list[str]:
    return text.casefold().split()


def error_rates(reference: str, prediction: str) -> tuple[float, float]:
    ref_chars = list(" ".join(normalized_tokens(reference)))
    pred_chars = list(" ".join(normalized_tokens(prediction)))
    ref_words = normalized_tokens(reference)
    pred_words = normalized_tokens(prediction)
    cer = edit_distance(ref_chars, pred_chars) / max(1, len(ref_chars))
    wer = edit_distance(ref_words, pred_words) / max(1, len(ref_words))
    return cer, wer


def benchmark_variant(module, version: str, image, reference: str, variant: str,
                      psm: int, confidence_threshold: float, repeats: int) -> dict:
    processed = module.preprocess(image)[variant]
    durations: list[float] = []
    final_data = None
    for _ in range(repeats):
        started = time.perf_counter()
        final_data = pytesseract.image_to_data(
            processed,
            config=f"--oem 3 --psm {psm}",
            output_type=Output.DICT,
        )
        durations.append(time.perf_counter() - started)

    words: list[dict] = []
    for index, raw_text in enumerate(final_data["text"]):
        word = raw_text.strip()
        try:
            score = float(final_data["conf"][index])
        except (ValueError, TypeError):
            continue
        if word and score >= 0:
            words.append({
                "text": word,
                "confidence": score,
                "block": int(final_data["block_num"][index]),
                "paragraph": int(final_data["par_num"][index]),
                "line": int(final_data["line_num"][index]),
            })
    accepted = [word for word in words if word["confidence"] >= confidence_threshold]
    lines: dict[tuple[int, int, int], list[str]] = {}
    for word in accepted:
        key = (word["block"], word["paragraph"], word["line"])
        lines.setdefault(key, []).append(word["text"])
    prediction = "\n".join(" ".join(parts) for parts in lines.values())
    cer, wer = error_rates(reference, prediction)
    return {
        "pipeline": version,
        "variant": variant,
        "cer": round(cer, 6),
        "wer": round(wer, 6),
        "processing_time_seconds_mean": round(sum(durations) / len(durations), 6),
        "processing_time_seconds_runs": [round(value, 6) for value in durations],
        "accepted_word_count": len(accepted),
        "mean_accepted_confidence": round(
            sum(word["confidence"] for word in accepted) / len(accepted), 1
        ) if accepted else 0.0,
        "recognized_text": prediction,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="samples/sample_document.png")
    parser.add_argument("--ground-truth", default=None)
    parser.add_argument("--confidence", type=float, default=80.0)
    parser.add_argument("--psm", type=int, default=6, choices=[3, 6, 7, 11])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", default="benchmarks/results")
    args = parser.parse_args()
    if not 0 <= args.confidence <= 100:
        parser.error("--confidence must be between 0 and 100")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    image_path = (ROOT / args.image).resolve()
    if not image_path.is_file():
        parser.error(f"Image not found: {image_path}")
    truth_path = Path(args.ground_truth) if args.ground_truth else (
        ROOT / "benchmarks" / "ground_truth" / f"{image_path.stem}.txt"
    )
    if not truth_path.is_absolute():
        truth_path = ROOT / truth_path
    if not truth_path.is_file():
        parser.error(f"Ground truth not found: {truth_path}; benchmark requires a separate reference file")

    baseline = load_baseline_module()
    baseline.configure_tesseract()
    upgraded.configure_tesseract()
    image = upgraded.load_image(image_path)
    reference = truth_path.read_text(encoding="utf-8")
    records = []
    for label, module in (("original_baseline", baseline), ("upgraded", upgraded)):
        for variant in ("grayscale", "adaptive_threshold"):
            records.append(benchmark_variant(
                module, label, image, reference, variant,
                args.psm, args.confidence, args.repeats,
            ))

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "image": str(image_path.relative_to(ROOT)) if image_path.is_relative_to(ROOT) else str(image_path),
        "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "ground_truth": str(truth_path.relative_to(ROOT)) if truth_path.is_relative_to(ROOT) else str(truth_path),
        "confidence_threshold": args.confidence,
        "psm": args.psm,
        "repeats": args.repeats,
        "tesseract_version": str(pytesseract.get_tesseract_version()).strip(),
        "methodology": "Case-folded whitespace tokenization; punctuation retained; CER/WER use Levenshtein edit distance.",
        "results": records,
    }
    (output_dir / "benchmark_results.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    fields = [
        "pipeline", "variant", "cer", "wer", "processing_time_seconds_mean",
        "accepted_word_count", "mean_accepted_confidence",
    ]
    with (output_dir / "benchmark_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: item[key] for key in fields} for item in records)

    print(f"Ground truth: {truth_path}")
    print(f"Tesseract: {metadata['tesseract_version']}")
    print("pipeline          variant              CER      WER      mean seconds  accepted  mean conf")
    for item in records:
        print(
            f"{item['pipeline']:<18} {item['variant']:<20} "
            f"{item['cer']:<8.4f} {item['wer']:<8.4f} "
            f"{item['processing_time_seconds_mean']:<13.4f} "
            f"{item['accepted_word_count']:<9} {item['mean_accepted_confidence']:.1f}%"
        )
    print(f"Reports saved to {output_dir}")


if __name__ == "__main__":
    main()
