from pathlib import Path

import joblib
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

WEEK2_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = WEEK2_DIR.parent

INPUT_FILE = PROJECT_DIR / "week-1" / "data" / "spam_cleaned.csv"

FEATURES_DIR = WEEK2_DIR / "features"

FEATURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Load cleaned Week 1 dataset
# ---------------------------------------------------------

print("=" * 70)
print("WEEK 2 - FEATURE EXTRACTION")
print("=" * 70)

print("\nLoading cleaned Week 1 dataset...")
print(f"Dataset path: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded successfully.")
print(f"Dataset shape: {df.shape}")


# ---------------------------------------------------------
# Validate required columns
# ---------------------------------------------------------

required_columns = ["label", "clean_text"]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


df = df.dropna(subset=["label", "clean_text"]).copy()

df["clean_text"] = df["clean_text"].astype(str).str.strip()

df = df[df["clean_text"] != ""]

print(f"Usable rows: {len(df)}")


# ---------------------------------------------------------
# Text and labels
# ---------------------------------------------------------

X_text = df["clean_text"]
y = df["label"]


# ---------------------------------------------------------
# Bag-of-Words
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("BAG-OF-WORDS FEATURE EXTRACTION")
print("=" * 70)

bow_vectorizer = CountVectorizer(
    min_df=2,
    ngram_range=(1, 2)
)

X_bow = bow_vectorizer.fit_transform(X_text)

print(f"Bag-of-Words matrix shape: {X_bow.shape}")
print(f"Number of BoW features: {len(bow_vectorizer.get_feature_names_out())}")


# ---------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TF-IDF FEATURE EXTRACTION")
print("=" * 70)

tfidf_vectorizer = TfidfVectorizer(
    min_df=2,
    ngram_range=(1, 2)
)

X_tfidf = tfidf_vectorizer.fit_transform(X_text)

print(f"TF-IDF matrix shape: {X_tfidf.shape}")
print(
    f"Number of TF-IDF features: "
    f"{len(tfidf_vectorizer.get_feature_names_out())}"
)


# ---------------------------------------------------------
# Save feature matrices
# ---------------------------------------------------------

print("\nSaving feature matrices...")

sparse.save_npz(
    FEATURES_DIR / "bow_features.npz",
    X_bow
)

sparse.save_npz(
    FEATURES_DIR / "tfidf_features.npz",
    X_tfidf
)

# Save labels
pd.DataFrame({
    "label": y.reset_index(drop=True)
}).to_csv(
    FEATURES_DIR / "labels.csv",
    index=False
)


# ---------------------------------------------------------
# Save vectorizers
# ---------------------------------------------------------

joblib.dump(
    bow_vectorizer,
    FEATURES_DIR / "bow_vectorizer.joblib"
)

joblib.dump(
    tfidf_vectorizer,
    FEATURES_DIR / "tfidf_vectorizer.joblib"
)


# ---------------------------------------------------------
# Display example features
# ---------------------------------------------------------

print("\nSample Bag-of-Words features:")

print(
    bow_vectorizer
    .get_feature_names_out()[:20]
)

print("\nSample TF-IDF features:")

print(
    tfidf_vectorizer
    .get_feature_names_out()[:20]
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE EXTRACTION COMPLETE")
print("=" * 70)

print(f"\nTotal emails: {len(df)}")
print(f"Bag-of-Words shape: {X_bow.shape}")
print(f"TF-IDF shape: {X_tfidf.shape}")

print("\nFiles saved to:")
print(FEATURES_DIR)