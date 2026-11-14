"""Command-line OCR: python main.py samples/sample_document.png"""
import argparse
import json
from pathlib import Path
from ocr_engine import load_image, recognize, save_image


def main():
    parser = argparse.ArgumentParser(description="DecodeLabs Project 4: OCR with image preprocessing")
    parser.add_argument("image", help="Input image path, e.g. samples/sample_document.png")
    parser.add_argument("--confidence", type=float, default=80.0, help="Minimum word confidence (0-100)")
    parser.add_argument("--psm", type=int, default=6, choices=[3,6,7,11], help="Tesseract page segmentation mode")
    args = parser.parse_args()
    image = load_image(args.image)
    result = recognize(image, min_confidence=args.confidence, psm=args.psm)
    outdir = Path("outputs")
    outdir.mkdir(exist_ok=True)
    save_image(outdir / "preprocessed.png", result["preprocessed"])
    save_image(outdir / "recognized_words.png", result["annotated"])
    (outdir / "extracted_text.txt").write_text(result["text"], encoding="utf-8")
    report = {k: v for k, v in result.items() if k not in ("preprocessed", "annotated")}
    (outdir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== Extracted text (confidence-filtered) ===")
    print(result["text"] or "[No words passed the confidence threshold]")
    print(f"\nRecognized words: {result['all_word_count']} | Passed >= {args.confidence:.0f}%: {result['accepted_word_count']}")
    print(f"Average confidence of accepted words: {result['mean_accepted_confidence']:.1f}%")
    print(f"Preprocessing selected: {result['variant']}")
    print(f"Saved results to {outdir.resolve()}")

if __name__ == "__main__":
    main()
