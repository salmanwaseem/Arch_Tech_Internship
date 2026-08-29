import re
import pandas as pd
from pathlib import Path


# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "data" / "spam_raw.csv"
OUTPUT_FILE = BASE_DIR / "data" / "spam_cleaned.csv"


def clean_email_text(text):
    """Clean and normalize raw email text."""

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Replace URLs
    text = re.sub(r"http\S+|www\.\S+", " url ", text)

    # Replace email addresses
    text = re.sub(
        r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
        " emailaddress ",
        text,
    )

    # Replace numbers
    text = re.sub(r"\b\d+\b", " number ", text)

    # Remove punctuation and special characters
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_dataset():
    print("Loading dataset...")
    df = pd.read_csv(INPUT_FILE)

    original_rows = len(df)

    print(f"Original shape: {df.shape}")

    # Keep only required columns
    df = df[["label", "text"]].copy()

    # Remove missing values
    df = df.dropna(subset=["label", "text"])

    # Keep only valid labels
    df = df[df["label"].isin([0, 1])]

    # Remove leading/trailing whitespace
    df["text"] = df["text"].astype(str).str.strip()

    # Remove empty emails
    df = df[df["text"] != ""]

    # Remove exact duplicate emails
    before_duplicates = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first")
    duplicates_removed = before_duplicates - len(df)

    # Clean text
    print("Cleaning email text...")
    df["clean_text"] = df["text"].apply(clean_email_text)

    # Remove emails that became empty after cleaning
    before_empty_clean = len(df)
    df = df[df["clean_text"] != ""]
    empty_after_cleaning_removed = before_empty_clean - len(df)

    # Remove duplicates created after text normalization
    before_clean_duplicates = len(df)
    df = df.drop_duplicates(subset=["clean_text"], keep="first")
    clean_duplicates_removed = before_clean_duplicates - len(df)

    # Reset row numbers
    df = df.reset_index(drop=True)

    # Save cleaned dataset
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n--- Cleaning Summary ---")
    print(f"Original rows: {original_rows}")
    print(f"Exact duplicates removed: {duplicates_removed}")
    print(f"Normalized duplicates removed: {clean_duplicates_removed}")
    print(f"Final rows: {len(df)}")
    print(f"Empty emails removed after cleaning: {empty_after_cleaning_removed}")

    print("\nLabel distribution:")
    print(df["label"].value_counts().sort_index())

    print("\nMissing values:")
    print(df.isna().sum())

    print(f"\nCleaned dataset saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    preprocess_dataset()