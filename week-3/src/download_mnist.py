"""Utility script to acquire and save the MNIST dataset for Week 3."""

import argparse
from pathlib import Path
import numpy as np
from sklearn.datasets import fetch_openml


def download_mnist(output_dir: Path, format: str = "npy", limit: int | None = 60000):
    """Download MNIST via OpenML and save in data directory.

    Args:
        output_dir: Target directory (e.g. week-3/data).
        format: Format to save: 'npy' (images.npy, labels.npy) or 'csv' (mnist_train.csv).
        limit: Number of training samples to save (default: 60,000).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching MNIST dataset (mnist_784)...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto")

    if limit is not None:
        X = X[:limit]
        y = y[:limit]

    X = X.astype(np.uint8)
    y = y.astype(np.int64)

    if format == "npy":
        images_path = output_dir / "images.npy"
        labels_path = output_dir / "labels.npy"
        np.save(images_path, X)
        np.save(labels_path, y)
        print(f"Saved {len(X):,} images to: {images_path}")
        print(f"Saved {len(y):,} labels to: {labels_path}")
    elif format == "csv":
        import pandas as pd
        csv_path = output_dir / "mnist_train.csv"
        columns = ["label"] + [f"pixel{i}" for i in range(784)]
        df = pd.DataFrame(np.column_stack([y, X]), columns=columns)
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(df):,} samples to CSV: {csv_path}")
    else:
        raise ValueError(f"Unknown format: {format}. Choose 'npy' or 'csv'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MNIST dataset for Week 3.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data"),
        help="Directory to save the dataset files",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["npy", "csv"],
        default="npy",
        help="Storage format ('npy' or 'csv')",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=60000,
        help="Number of training samples (default: 60000)",
    )
    args = parser.parse_args()
    download_mnist(Path(args.output_dir), format=args.format, limit=args.samples)
