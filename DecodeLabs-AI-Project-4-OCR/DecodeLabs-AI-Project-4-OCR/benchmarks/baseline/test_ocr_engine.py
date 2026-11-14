import unittest
import numpy as np
from ocr_engine import preprocess, recognize

class OCRTests(unittest.TestCase):
    def test_preprocess_images(self):
        img = np.full((100, 150, 3), 255, dtype=np.uint8)
        variants = preprocess(img)
        self.assertEqual(set(variants), {"grayscale", "adaptive_threshold"})
        for variant in variants.values():
            self.assertEqual(variant.shape, (100, 150))

    def test_invalid_confidence(self):
        img = np.zeros((40, 40, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            recognize(img, min_confidence=101)

    def test_invalid_psm(self):
        img = np.zeros((40, 40, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            recognize(img, psm=5)

if __name__ == "__main__":
    unittest.main()
