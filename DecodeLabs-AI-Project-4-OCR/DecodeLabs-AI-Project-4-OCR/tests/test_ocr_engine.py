import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from PIL import Image

from ocr_engine import (
    decode_image_bytes,
    load_image,
    normalize_image,
    preprocess,
    recognize,
    save_image,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples" / "sample_document.png"
HAS_TESSERACT = bool(
    shutil.which("tesseract")
    or (os.environ.get("TESSERACT_EXE") and Path(os.environ["TESSERACT_EXE"]).is_file())
    or Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe").is_file()
    or Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe").is_file()
)


def encoded_image(image: np.ndarray, extension: str = ".png") -> bytes:
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise AssertionError("Test image encoding failed")
    return encoded.tobytes()


def fake_ocr_data():
    return {
        "text": ["kept", "rejected"],
        "conf": ["91.5", "40.0"],
        "left": [8, 30],
        "top": [9, 15],
        "width": [18, 25],
        "height": [12, 14],
        "block_num": [1, 1],
        "par_num": [1, 1],
        "line_num": [1, 1],
    }


class LegacyCompatibilityTests(unittest.TestCase):
    def test_preprocess_images(self):
        image = np.full((100, 150, 3), 255, dtype=np.uint8)
        variants = preprocess(image)
        self.assertEqual(set(variants), {"grayscale", "adaptive_threshold"})
        for variant in variants.values():
            self.assertEqual(variant.shape, (100, 150))

    def test_original_preprocessing_is_pixel_compatible(self):
        image = np.random.default_rng(15).integers(0, 256, (100, 150, 3), dtype=np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        smooth = cv2.GaussianBlur(gray, (3, 3), 0)
        threshold = cv2.adaptiveThreshold(
            smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 11,
        )
        result = preprocess(image)
        np.testing.assert_array_equal(result["grayscale"], gray)
        np.testing.assert_array_equal(result["adaptive_threshold"], threshold)

    def test_invalid_confidence(self):
        image = np.zeros((40, 40, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            recognize(image, min_confidence=101)

    def test_invalid_psm(self):
        image = np.zeros((40, 40, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            recognize(image, psm=5)

    def test_recognize_preserves_old_result_fields_and_filters_words(self):
        image = np.full((50, 80, 3), 255, dtype=np.uint8)
        with mock.patch("ocr_engine.configure_tesseract"), mock.patch(
            "ocr_engine.pytesseract.image_to_data", side_effect=[fake_ocr_data(), fake_ocr_data()]
        ):
            result = recognize(image, min_confidence=80, psm=6)
        self.assertTrue({
            "text", "words", "all_word_count", "accepted_word_count",
            "mean_accepted_confidence", "variant", "min_confidence",
            "preprocessed", "annotated",
        }.issubset(result))
        self.assertEqual(result["text"], "kept")
        self.assertEqual(result["all_word_count"], 2)
        self.assertEqual(result["accepted_word_count"], 1)
        self.assertGreaterEqual(result["processing_time_seconds"], 0)
        self.assertEqual(result["annotated"].shape, image.shape)


class ImageInputTests(unittest.TestCase):
    def test_color_input_is_unchanged(self):
        image = np.zeros((30, 40, 3), dtype=np.uint8)
        self.assertIs(normalize_image(image), image)

    def test_grayscale_normalizes_to_bgr(self):
        image = np.full((30, 40), 127, dtype=np.uint8)
        result = normalize_image(image)
        self.assertEqual(result.shape, (30, 40, 3))
        np.testing.assert_array_equal(result[:, :, 0], image)

    def test_transparent_pixels_composite_on_white(self):
        image = np.zeros((10, 12, 4), dtype=np.uint8)
        image[:, :, :3] = (0, 0, 255)
        image[:, :, 3] = 0
        result = normalize_image(image)
        self.assertEqual(result.shape, (10, 12, 3))
        self.assertTrue(np.all(result == 255))

    def test_empty_invalid_and_wrong_channel_inputs_rejected(self):
        cases = [
            (np.array([], dtype=np.uint8), "empty"),
            (np.zeros((4, 4, 2), dtype=np.uint8), "grayscale, BGR, or BGRA"),
            (np.zeros((4, 4), dtype=np.float32), "uint8"),
            (None, "NumPy array"),
        ]
        for image, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                normalize_image(image)

    def test_oversized_dimensions_rejected_before_processing(self):
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        with mock.patch.dict(os.environ, {"VISIONTEXT_MAX_IMAGE_PIXELS": "10"}):
            with self.assertRaisesRegex(ValueError, "exceed the limit"):
                normalize_image(image)

    def test_configurable_upload_size_limit(self):
        with mock.patch.dict(os.environ, {"VISIONTEXT_MAX_UPLOAD_MB": "1"}):
            with self.assertRaisesRegex(ValueError, "upload limit"):
                decode_image_bytes(b"x" * (1024 * 1024 + 1))

    def test_invalid_limit_configuration_is_reported(self):
        with mock.patch.dict(os.environ, {"VISIONTEXT_MAX_IMAGE_PIXELS": "0"}):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                normalize_image(np.zeros((2, 2, 3), dtype=np.uint8))

    def test_encoded_grayscale_and_bgra_pngs_load_as_bgr(self):
        gray = np.full((20, 30), 88, dtype=np.uint8)
        gray_result = decode_image_bytes(encoded_image(gray))
        self.assertEqual(gray_result.shape, (20, 30, 3))
        self.assertTrue(np.all(gray_result == 88))

        bgra = np.full((15, 25, 4), 255, dtype=np.uint8)
        bgra[:, :, :3] = (0, 0, 0)
        bgra[:, :, 3] = 0
        rgba_result = decode_image_bytes(encoded_image(bgra))
        self.assertEqual(rgba_result.shape, (15, 25, 3))
        self.assertTrue(np.all(rgba_result == 255))

    def test_jpeg_upload_decodes_to_color_image(self):
        image = np.full((22, 31, 3), (30, 120, 220), dtype=np.uint8)
        loaded = decode_image_bytes(encoded_image(image, ".jpg"))
        self.assertEqual(loaded.shape, (22, 31, 3))
        self.assertEqual(loaded.dtype, np.uint8)

    def test_empty_corrupt_and_unsupported_image_bytes_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            decode_image_bytes(b"")
        with self.assertRaisesRegex(ValueError, "valid, readable"):
            decode_image_bytes(b"not an image")
        bmp = encoded_image(np.zeros((8, 8, 3), dtype=np.uint8), ".bmp")
        with self.assertRaisesRegex(ValueError, "Unsupported image format"):
            decode_image_bytes(bmp)

    def test_decoded_image_dimensions_checked_from_header(self):
        png = encoded_image(np.zeros((4, 4, 3), dtype=np.uint8))
        with mock.patch.dict(os.environ, {"VISIONTEXT_MAX_IMAGE_PIXELS": "10"}):
            with self.assertRaisesRegex(ValueError, "exceed the limit"):
                decode_image_bytes(png)

    def test_load_image_checks_path_extension_and_saves_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.png"
            path.write_bytes(encoded_image(np.full((12, 16, 3), 200, dtype=np.uint8)))
            loaded = load_image(path)
            self.assertEqual(loaded.shape, (12, 16, 3))
            output = Path(tmp) / "nested" / "saved.png"
            save_image(output, loaded)
            with Image.open(output) as saved:
                self.assertEqual(saved.size, (16, 12))
            unsupported = Path(tmp) / "sample.gif"
            unsupported.write_bytes(path.read_bytes())
            with self.assertRaisesRegex(ValueError, "Unsupported image format"):
                load_image(unsupported)

    def test_missing_tesseract_has_actionable_error(self):
        with mock.patch.dict(os.environ, {"TESSERACT_EXE": ""}), mock.patch(
            "ocr_engine.shutil.which", return_value=None
        ), mock.patch("ocr_engine.Path.is_file", return_value=False):
            from ocr_engine import configure_tesseract
            with self.assertRaisesRegex(RuntimeError, "Tesseract OCR engine not found"):
                configure_tesseract()


@unittest.skipUnless(HAS_TESSERACT, "Tesseract executable is not installed")
class OCRIntegrationTests(unittest.TestCase):
    def test_known_sample_phrases_and_bounding_box_coordinates(self):
        image = load_image(SAMPLE)
        result = recognize(image, min_confidence=80, psm=6)
        for phrase in (
            "Optical Character Recognition",
            "Python + OpenCV + Tesseract",
            "This document contains sample printed text.",
            "Recognition should be readable and clear.",
            "Confidence filtering: 80 percent",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, result["text"])
        height, width = image.shape[:2]
        self.assertEqual(result["annotated"].shape, image.shape)
        for word in result["words"]:
            self.assertGreaterEqual(word["left"], 0)
            self.assertGreaterEqual(word["top"], 0)
            self.assertGreater(word["width"], 0)
            self.assertGreater(word["height"], 0)
            self.assertLessEqual(word["left"] + word["width"], width + 1)
            self.assertLessEqual(word["top"] + word["height"], height + 1)

    def test_blank_image_returns_no_accepted_words(self):
        result = recognize(np.full((100, 150, 3), 255, dtype=np.uint8), min_confidence=80)
        self.assertEqual(result["accepted_word_count"], 0)
        self.assertEqual(result["text"], "")

    def test_cli_processes_sample_and_writes_text_report_and_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            completed = subprocess.run(
                [sys.executable, str(ROOT / "main.py"), str(SAMPLE)],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=45,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Optical Character Recognition", completed.stdout)
            output = Path(tmp) / "outputs"
            for name in ("extracted_text.txt", "report.json", "preprocessed.png", "recognized_words.png"):
                self.assertTrue((output / name).is_file(), name)
            self.assertIn("Confidence filtering: 80 percent", (output / "extracted_text.txt").read_text())

    def test_streamlit_dashboard_module_imports(self):
        completed = subprocess.run(
            [sys.executable, "-c", "import app"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
