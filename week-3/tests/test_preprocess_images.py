"""Unit and validation tests for Week 3 MNIST preprocessing."""

from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

# Add src to sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocess_images import (
    normalize_pixels,
    reshape_images,
    split_dataset,
    preprocess_and_save,
)


class TestNormalizePixels(unittest.TestCase):
    """Tests for pixel normalization."""

    def test_normalize_standard_range(self):
        """Test normalizing uint8/float range (0-255)."""
        images = np.array([[[0.0, 127.5], [255.0, 64.0]]], dtype=np.float32)
        normalized = normalize_pixels(images)
        self.assertTrue(np.isclose(normalized.max(), 1.0))
        self.assertTrue(np.isclose(normalized.min(), 0.0))
        self.assertTrue(np.isclose(normalized[0, 0, 1], 127.5 / 255.0))

    def test_normalize_already_normalized(self):
        """Test that data already in 0-1 range is preserved."""
        images = np.array([[[0.0, 0.5], [1.0, 0.25]]], dtype=np.float32)
        normalized = normalize_pixels(images)
        self.assertTrue(np.allclose(normalized, images))

    def test_normalize_empty_raises(self):
        """Empty input should raise ValueError."""
        with self.assertRaises(ValueError):
            normalize_pixels(np.array([]))

    def test_normalize_non_positive_raises(self):
        """All-zero or negative pixels should raise ValueError."""
        with self.assertRaises(ValueError):
            normalize_pixels(np.zeros((5, 28, 28)))


class TestReshapeImages(unittest.TestCase):
    """Tests for image reshaping."""

    def test_reshape_flatten_from_2d(self):
        """Flattening an already-flat 2D array."""
        images = np.zeros((10, 784))
        result = reshape_images(images, format="flatten")
        self.assertEqual(result.shape, (10, 784))

    def test_reshape_flatten_from_3d(self):
        """Flattening a 3D (samples, 28, 28) array."""
        images = np.zeros((10, 28, 28))
        result = reshape_images(images, format="flatten")
        self.assertEqual(result.shape, (10, 784))

    def test_reshape_channels_last(self):
        """Reshaping to channels-last tensor (samples, 28, 28, 1)."""
        images = np.zeros((10, 784))
        result = reshape_images(images, format="channels_last")
        self.assertEqual(result.shape, (10, 28, 28, 1))

    def test_reshape_channels_first(self):
        """Reshaping to channels-first tensor (samples, 1, 28, 28)."""
        images = np.zeros((10, 784))
        result = reshape_images(images, format="channels_first")
        self.assertEqual(result.shape, (10, 1, 28, 28))

    def test_reshape_non_square_raises(self):
        """Non-square 2D feature count should raise ValueError."""
        with self.assertRaises(ValueError):
            reshape_images(np.zeros((10, 780)), format="flatten")

    def test_reshape_invalid_format_raises(self):
        """Unknown format option should raise ValueError."""
        with self.assertRaises(ValueError):
            reshape_images(np.zeros((10, 28, 28)), format="invalid")


class TestSplitDataset(unittest.TestCase):
    """Tests for dataset splitting."""

    def test_stratified_split(self):
        """Test 80/20 split with sufficient samples per class."""
        images = np.zeros((100, 784))
        labels = np.repeat(np.arange(10), 10)
        x_tr, x_te, y_tr, y_te = split_dataset(images, labels, test_size=0.2, random_state=42)
        self.assertEqual(len(x_tr), 80)
        self.assertEqual(len(x_te), 20)
        self.assertEqual(len(y_tr), 80)
        self.assertEqual(len(y_te), 20)
        self.assertEqual(len(np.unique(y_te)), 10)

    def test_split_small_dataset_fallback(self):
        """Test that small datasets fallback gracefully without crashing."""
        images = np.zeros((10, 784))
        labels = np.arange(10)
        x_tr, x_te, y_tr, y_te = split_dataset(images, labels, test_size=0.2, random_state=42)
        self.assertEqual(len(x_tr), 8)
        self.assertEqual(len(x_te), 2)

    def test_length_mismatch_raises(self):
        """Mismatched sample counts between images and labels should raise ValueError."""
        with self.assertRaises(ValueError):
            split_dataset(np.zeros((10, 784)), np.zeros(5))


class TestPreprocessAndSave(unittest.TestCase):
    """End-to-end preprocessing and saving tests."""

    def test_pipeline_and_output_file(self):
        """Test full pipeline creates valid npz with expected keys and shapes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "mnist_processed.npz"
            images = np.random.randint(1, 256, size=(50, 28, 28)).astype(np.float32)
            labels = np.repeat(np.arange(10), 5)

            saved_path = preprocess_and_save(images, labels, out_file, reshape_format="flatten")
            self.assertTrue(saved_path.exists())

            with np.load(saved_path) as loaded:
                self.assertIn("x_train", loaded)
                self.assertIn("x_test", loaded)
                self.assertIn("y_train", loaded)
                self.assertIn("y_test", loaded)
                self.assertEqual(loaded["x_train"].shape, (40, 784))
                self.assertEqual(loaded["x_test"].shape, (10, 784))
                self.assertLessEqual(loaded["x_train"].max(), 1.0)
                self.assertGreaterEqual(loaded["x_train"].min(), 0.0)


if __name__ == "__main__":
    unittest.main()
