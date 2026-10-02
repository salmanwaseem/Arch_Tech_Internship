"""Reusable preprocessing helpers for the MNIST digit-recognition task."""

from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split


def normalize_pixels(images: np.ndarray) -> np.ndarray:
    """Convert pixel values to floating-point values between 0 and 1."""
    images = np.asarray(images, dtype=np.float32)
    if images.size == 0:
        raise ValueError("images must contain at least one sample")

    maximum = float(np.nanmax(images))
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError("images must contain finite, positive pixel values")

    # MNIST pixels normally use 0-255; this also handles already-normalized data.
    if maximum > 1.0:
        images = images / 255.0
    return np.clip(images, 0.0, 1.0)


def reshape_images(images: np.ndarray, format: str = "flatten") -> np.ndarray:
    """Reshape images for a model as flattened vectors or image tensors.

    Supported formats are ``flatten`` (samples, pixels), ``channels_last``
    (samples, height, width, channels), and ``channels_first``.
    """
    images = np.asarray(images)
    if images.ndim == 2:
        sample_count, pixel_count = images.shape
        side = int(np.sqrt(pixel_count))
        if side * side != pixel_count:
            raise ValueError("2-D images must contain a square number of pixels")
        images = images.reshape(sample_count, side, side)
    elif images.ndim == 3:
        pass
    elif images.ndim == 4 and images.shape[-1] == 1:
        images = images[..., 0]
    else:
        raise ValueError("images must have shape (samples, pixels), (samples, height, width), or a single channel")

    if format == "flatten":
        return images.reshape(images.shape[0], -1)
    if format == "channels_last":
        return images[..., np.newaxis]
    if format == "channels_first":
        return images[:, np.newaxis, ...]
    raise ValueError("format must be 'flatten', 'channels_last', or 'channels_first'")


def split_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split images and labels into reproducible training and testing sets."""
    images = np.asarray(images)
    labels = np.asarray(labels)
    if len(images) != len(labels):
        raise ValueError("images and labels must contain the same number of samples")

    unique_labels = np.unique(labels)
    test_count = len(labels) * test_size if 0.0 < test_size < 1.0 else int(test_size)
    if len(unique_labels) > 0 and test_count < len(unique_labels):
        return train_test_split(
            images,
            labels,
            test_size=test_size,
            random_state=random_state,
        )

    try:
        return train_test_split(
            images,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )
    except ValueError:
        return train_test_split(
            images,
            labels,
            test_size=test_size,
            random_state=random_state,
        )


def preprocess_and_save(
    images: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
    reshape_format: str = "flatten",
) -> Path:
    """Normalize, reshape, split, and save processed MNIST arrays as NPZ."""
    normalized = normalize_pixels(images)
    reshaped = reshape_images(normalized, format=reshape_format)
    x_train, x_test, y_train, y_test = split_dataset(reshaped, labels)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
    )
    return output_path
