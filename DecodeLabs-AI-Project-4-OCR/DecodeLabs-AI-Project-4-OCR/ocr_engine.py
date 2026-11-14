"""Safe image loading and confidence-filtered OCR with the original pipeline."""
from __future__ import annotations

import os
import shutil
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytesseract
from PIL import Image, UnidentifiedImageError
from pytesseract import Output

DEFAULT_MAX_UPLOAD_MB = 15
DEFAULT_MAX_IMAGE_PIXELS = 25_000_000
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
SUPPORTED_PIL_FORMATS = {"PNG", "JPEG"}


def get_image_limits() -> tuple[int, int]:
    """Return (maximum upload bytes, maximum decoded pixels) from environment settings."""
    values = (
        ("VISIONTEXT_MAX_UPLOAD_MB", DEFAULT_MAX_UPLOAD_MB),
        ("VISIONTEXT_MAX_IMAGE_PIXELS", DEFAULT_MAX_IMAGE_PIXELS),
    )
    parsed: list[int] = []
    for name, default in values:
        raw = os.environ.get(name, str(default)).strip()
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be a positive integer.") from exc
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
        parsed.append(value)
    return parsed[0] * 1024 * 1024, parsed[1]


def configure_tesseract() -> str:
    """Locate Tesseract, including common Windows installation paths."""
    explicit = os.environ.get("TESSERACT_EXE", "").strip()
    candidates = [
        explicit,
        shutil.which("tesseract"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for exe in candidates:
        if exe and Path(exe).is_file():
            pytesseract.pytesseract.tesseract_cmd = str(exe)
            return str(exe)
    raise RuntimeError(
        "Tesseract OCR engine not found. Install Tesseract separately from pip, "
        "add it to PATH, or set TESSERACT_EXE to the executable path."
    )


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Validate an image and normalize grayscale/BGRA inputs to 8-bit BGR."""
    if not isinstance(img, np.ndarray):
        raise ValueError("Image must be a NumPy array.")
    if img.size == 0:
        raise ValueError("Image is empty.")
    if img.dtype != np.uint8:
        raise ValueError("Image must use 8-bit unsigned pixels (uint8).")
    if img.ndim == 2:
        height, width = img.shape
        channels = 1
    elif img.ndim == 3 and img.shape[2] in (3, 4):
        height, width, channels = img.shape
    else:
        raise ValueError("Image must be grayscale, BGR, or BGRA.")
    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be greater than zero.")
    _, max_pixels = get_image_limits()
    if height * width > max_pixels:
        raise ValueError(
            f"Image dimensions ({width} × {height}) exceed the limit of "
            f"{max_pixels:,} pixels. Set VISIONTEXT_MAX_IMAGE_PIXELS to change it."
        )
    if channels == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if channels == 4:
        # Composite transparency onto white; transparent pixels should not become black.
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        color = img[:, :, :3].astype(np.float32)
        return np.clip(color * alpha + 255.0 * (1.0 - alpha), 0, 255).astype(np.uint8)
    return img


def decode_image_bytes(data: bytes | bytearray | memoryview) -> np.ndarray:
    """Decode a bounded PNG/JPEG upload or file buffer and return a BGR image."""
    try:
        raw = bytes(data)
    except (TypeError, ValueError) as exc:
        raise ValueError("Image data must be bytes-like.") from exc
    if not raw:
        raise ValueError("The uploaded image is empty.")
    max_bytes, max_pixels = get_image_limits()
    if len(raw) > max_bytes:
        limit_mb = max_bytes / (1024 * 1024)
        raise ValueError(f"Image file is larger than the {limit_mb:g} MB upload limit.")
    try:
        with Image.open(BytesIO(raw)) as image_header:
            image_format = image_header.format
            width, height = image_header.size
            if image_format not in SUPPORTED_PIL_FORMATS:
                raise ValueError("Unsupported image format. Please use PNG, JPG, or JPEG.")
            if width <= 0 or height <= 0:
                raise ValueError("Image dimensions must be greater than zero.")
            if width * height > max_pixels:
                raise ValueError(
                    f"Image dimensions ({width} × {height}) exceed the limit of "
                    f"{max_pixels:,} pixels. Set VISIONTEXT_MAX_IMAGE_PIXELS to change it."
                )
            image_header.verify()
    except Image.DecompressionBombError as exc:
        raise ValueError("Image dimensions exceed Pillow's safe decoder limit.") from exc
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ValueError("The file is not a valid, readable PNG or JPEG image.") from exc
    decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise ValueError("The file is not a valid, readable PNG or JPEG image.")
    return normalize_image(decoded)


def load_image(path: str | Path) -> np.ndarray:
    """Load a PNG/JPG/JPEG path, checking file size and dimensions before decoding."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported image format. Please use PNG, JPG, or JPEG.")
    max_bytes, _ = get_image_limits()
    try:
        file_size = path.stat().st_size
    except OSError as exc:
        raise ValueError(f"Cannot read image file: {path}") from exc
    if file_size > max_bytes:
        limit_mb = max_bytes / (1024 * 1024)
        raise ValueError(f"Image file is larger than the {limit_mb:g} MB upload limit.")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Cannot read image file: {path}") from exc
    return decode_image_bytes(raw)


def preprocess(img: np.ndarray) -> dict[str, np.ndarray]:
    """Run the original grayscale and adaptive-threshold preprocessing unchanged."""
    bgr = normalize_image(img)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    threshold = cv2.adaptiveThreshold(
        smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    return {"grayscale": gray, "adaptive_threshold": threshold}


def recognize(img: np.ndarray, *, min_confidence: float = 80.0, psm: int = 6) -> dict[str, Any]:
    """Recognize text while retaining the original result keys and OCR selection behavior."""
    if not 0 <= min_confidence <= 100:
        raise ValueError("min_confidence must be between 0 and 100")
    if psm not in (3, 6, 7, 11):
        raise ValueError("psm must be 3, 6, 7, or 11")
    configure_tesseract()
    original = normalize_image(img)
    started = time.perf_counter()
    variants = preprocess(original)
    results = []
    for label, processed in variants.items():
        data = pytesseract.image_to_data(
            processed, config=f"--oem 3 --psm {psm}", output_type=Output.DICT
        )
        words = []
        for i, text in enumerate(data["text"]):
            word = text.strip()
            try:
                confidence = float(data["conf"][i])
            except (ValueError, TypeError):
                continue
            if not word or confidence < 0:
                continue
            words.append(
                {
                    "text": word,
                    "confidence": round(confidence, 1),
                    "left": int(data["left"][i]),
                    "top": int(data["top"][i]),
                    "width": int(data["width"][i]),
                    "height": int(data["height"][i]),
                    "block": int(data["block_num"][i]),
                    "paragraph": int(data["par_num"][i]),
                    "line": int(data["line_num"][i]),
                }
            )
        accepted = [word for word in words if word["confidence"] >= min_confidence]
        score = sum(word["confidence"] for word in accepted) / len(accepted) if accepted else 0.0
        results.append((len(accepted), score, label, processed, words, accepted))
    # Preserve the existing preference: more accepted words, then higher mean confidence.
    count, mean, name, processed, all_words, accepted = max(results, key=lambda item: (item[0], item[1]))
    lines: dict[tuple[int, int, int], list[str]] = {}
    for word in accepted:
        key = (word["block"], word["paragraph"], word["line"])
        lines.setdefault(key, []).append(word["text"])
    text = "\n".join(" ".join(items) for items in lines.values())
    annotated = original.copy()
    height, width = original.shape[:2]
    for word in accepted:
        x, y = word["left"], word["top"]
        box_width, box_height = word["width"], word["height"]
        if box_width <= 0 or box_height <= 0:
            continue
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(width - 1, x + box_width), min(height - 1, y + box_height)
        if x1 <= x2 and y1 <= y2:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (25, 180, 25), 2)
    processing_time = time.perf_counter() - started
    return {
        "text": text,
        "words": accepted,
        "all_word_count": len(all_words),
        "accepted_word_count": len(accepted),
        "mean_accepted_confidence": round(mean, 1),
        "variant": name,
        "min_confidence": min_confidence,
        "preprocessed": processed,
        "annotated": annotated,
        "processing_time_seconds": round(processing_time, 4),
    }


def save_image(path: str | Path, img: np.ndarray) -> None:
    """Save an image to a path, creating its parent directory if required."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
        raise ValueError(f"Unsupported output image extension: {ext}")
    ok, encoded = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"Could not encode output image using extension: {ext}")
    encoded.tofile(str(path))
