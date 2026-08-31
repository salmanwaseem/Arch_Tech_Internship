import re

import joblib

from pathlib import Path


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

WEEK2_DIR = Path(__file__).resolve().parents[1]

MODEL_FILE = (
    WEEK2_DIR
    / "models"
    / "spam_classifier.joblib"
)


# ---------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------

def clean_email_text(text):
    """
    Clean email text using the same preprocessing
    approach used for the training dataset.
    """

    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove HTML
    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    # Replace URLs
    text = re.sub(
        r"http\S+|www\.\S+",
        " url ",
        text
    )

    # Replace email addresses
    text = re.sub(
        r"\b[a-zA-Z0-9._%+-]+"
        r"@[a-zA-Z0-9.-]+"
        r"\.[a-zA-Z]{2,}\b",
        " emailaddress ",
        text
    )

    # Replace numbers
    text = re.sub(
        r"\b\d+\b",
        " number ",
        text
    )

    # Remove punctuation and special characters
    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    # Remove extra whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

def load_model():

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}"
        )

    print("=" * 70)
    print("SPAM EMAIL CLASSIFIER")
    print("=" * 70)

    print("\nLoading trained model...")
    print(f"Model: {MODEL_FILE}")

    model = joblib.load(MODEL_FILE)

    print("Model loaded successfully.")

    return model


# ---------------------------------------------------------
# Predict sample emails
# ---------------------------------------------------------

def test_sample_emails(model):

    sample_emails = [

        # Expected spam
        (
            "Congratulations! You have won a free cash prize. "
            "Click here now to claim your reward."
        ),

        # Expected ham
        (
            "Hi Sarah, please find attached the meeting notes "
            "from today's project discussion."
        ),

        # Expected spam
        (
            "URGENT! You have been selected for a special prize. "
            "Claim your reward before the offer expires."
        ),

        # Expected ham
        (
            "Can we reschedule tomorrow's team meeting "
            "to 3 PM?"
        ),

        # Expected spam
        (
            "You are our lucky winner! Click this link "
            "to receive your free gift immediately."
        ),

        # Expected ham
        (
            "Please review the latest version of the project "
            "report and send your feedback."
        )
    ]

    print("\n" + "=" * 70)
    print("SAMPLE EMAIL PREDICTIONS")
    print("=" * 70)

    for number, email in enumerate(
        sample_emails,
        start=1
    ):

        cleaned_email = clean_email_text(email)

        prediction = model.predict(
            [cleaned_email]
        )[0]

        label = (
            "SPAM"
            if prediction == 1
            else "HAM"
        )

        print("\n" + "-" * 70)

        print(f"Email {number}:")
        print(email)

        print(f"\nCleaned text:")
        print(cleaned_email)

        print(f"\nPrediction: {label}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    model = load_model()

    test_sample_emails(model)

    print("\n" + "=" * 70)
    print("PREDICTION TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()