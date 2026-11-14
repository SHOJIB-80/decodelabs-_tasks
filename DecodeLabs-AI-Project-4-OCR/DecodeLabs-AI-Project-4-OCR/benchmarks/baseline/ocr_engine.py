"""Preprocess images and perform confidence-filtered OCR with Tesseract."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from pytesseract import Output


def configure_tesseract() -> str:
    """Locate Tesseract, including common Windows installation paths."""
    explicit = os.environ.get("TESSERACT_EXE", "").strip()
    candidates = [explicit, shutil.which("tesseract"),
                  r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                  r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]
    for exe in candidates:
        if exe and Path(exe).is_file():
            pytesseract.pytesseract.tesseract_cmd = str(exe)
            return str(exe)
    raise RuntimeError(
        "Tesseract OCR engine not found. Install Tesseract on Windows and add "
        "C:\\Program Files\\Tesseract-OCR to PATH, or set the TESSERACT_EXE environment variable. "
        "Installing pytesseract via pip does not install the OCR engine."
    )


def load_image(path: str | Path) -> np.ndarray:
    """Load images from Unicode paths, including Windows folders with spaces."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    raw = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot open image: {path}")
    return img


def preprocess(img: np.ndarray) -> dict[str, np.ndarray]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    threshold = cv2.adaptiveThreshold(smooth, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
    return {"grayscale": gray, "adaptive_threshold": threshold}


def recognize(img: np.ndarray, *, min_confidence: float = 80.0, psm: int = 6) -> dict:
    if not 0 <= min_confidence <= 100:
        raise ValueError("min_confidence must be between 0 and 100")
    if psm not in (3, 6, 7, 11):
        raise ValueError("psm must be 3, 6, 7, or 11")
    configure_tesseract()
    variants = preprocess(img)
    results = []
    for label, processed in variants.items():
        data = pytesseract.image_to_data(processed, config=f"--oem 3 --psm {psm}", output_type=Output.DICT)
        words = []
        for i, text in enumerate(data["text"]):
            word = text.strip()
            try:
                confidence = float(data["conf"][i])
            except (ValueError, TypeError):
                continue
            if not word or confidence < 0:
                continue
            words.append({"text": word, "confidence": round(confidence, 1),
                          "left": int(data["left"][i]), "top": int(data["top"][i]),
                          "width": int(data["width"][i]), "height": int(data["height"][i]),
                          "block": int(data["block_num"][i]),
                          "paragraph": int(data["par_num"][i]), "line": int(data["line_num"][i])})
        accepted = [word for word in words if word["confidence"] >= min_confidence]
        score = (sum(w["confidence"] for w in accepted) / len(accepted)) if accepted else 0.0
        results.append((len(accepted), score, label, processed, words, accepted))
    # Prefer more confidently recognized words, then higher mean confidence.
    count, mean, name, processed, all_words, accepted = max(results, key=lambda x: (x[0], x[1]))
    # Preserve approximate line grouping from Tesseract results.
    lines = {}
    for w in accepted:
        key = (w["block"], w["paragraph"], w["line"])
        lines.setdefault(key, []).append(w["text"])
    text = "\n".join(" ".join(items) for items in lines.values())
    annotated = img.copy()
    for word in accepted:
        x, y, w, h = word["left"], word["top"], word["width"], word["height"]
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (25, 180, 25), 2)
    return {"text": text, "words": accepted, "all_word_count": len(all_words),
            "accepted_word_count": len(accepted), "mean_accepted_confidence": round(mean, 1),
            "variant": name, "min_confidence": min_confidence,
            "preprocessed": processed, "annotated": annotated}


def save_image(path: str | Path, img: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    ok, encoded = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"Unsupported output image extension: {ext}")
    encoded.tofile(str(path))
