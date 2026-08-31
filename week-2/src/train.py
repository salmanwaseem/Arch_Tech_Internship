from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# ============================================================
# PROJECT PATHS
# ============================================================


WEEK2_DIR = Path(__file__).resolve().parents[1]


PROJECT_ROOT = WEEK2_DIR.parent


DATA_FILE = (
    PROJECT_ROOT
    / "week-1"
    / "data"
    / "spam_cleaned.csv"
)


MODEL_DIR = WEEK2_DIR / "models"
RESULTS_DIR = WEEK2_DIR / "results"


MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(model_name, stage, model, X_test, y_test):
    """
    Evaluate a trained model using:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - Classification report
    - Confusion matrix
    """

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    print("\n" + "=" * 70)
    print(f"{model_name} - {stage}")
    print("=" * 70)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Ham", "Spam"],
            zero_division=0,
            digits=4,
        )
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=[0, 1],
        display_labels=["Ham", "Spam"],
    )

    plt.title(
        f"{model_name} - {stage} Confusion Matrix"
    )

    plt.tight_layout()


    file_name = (
        f"{model_name}_{stage}_confusion_matrix"
        .lower()
        .replace(" ", "_")
    )

    confusion_matrix_path = (
        RESULTS_DIR
        / f"{file_name}.png"
    )

    plt.savefig(
        confusion_matrix_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Confusion matrix saved to: "
        f"{confusion_matrix_path}"
    )

    return {
        "Model": model_name,
        "Stage": stage,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
    }


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main():

    # ========================================================
    # 1. LOAD CLEANED WEEK 1 DATASET
    # ========================================================

    print("=" * 70)
    print("WEEK 2 - SPAM EMAIL CLASSIFICATION")
    print("=" * 70)

    print("\nLoading cleaned Week 1 dataset...")
    print(f"Dataset path: {DATA_FILE}")

    # Check that Week 1 preprocessing has been performed
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "\nCleaned dataset was not found.\n\n"
            f"Expected file:\n{DATA_FILE}\n\n"
            "Run the Week 1 preprocessing script first:\n"
            "python week-1/src/preprocess.py"
        )

    df = pd.read_csv(DATA_FILE)

    print("\nDataset loaded successfully.")
    print(f"Dataset shape: {df.shape}")

    print("\nDataset columns:")
    print(df.columns.tolist())


    # ========================================================
    # 2. VALIDATE REQUIRED COLUMNS
    # ========================================================

    required_columns = {
        "label",
        "clean_text",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Required columns are missing from the dataset: "
            f"{missing_columns}"
        )


    # ========================================================
    # 3. REMOVE ANY UNEXPECTED MISSING VALUES
    # ========================================================

    df = df.dropna(
        subset=[
            "label",
            "clean_text",
        ]
    ).copy()

    df["clean_text"] = (
        df["clean_text"]
        .astype(str)
        .str.strip()
    )

  
    df = df[
        df["clean_text"] != ""
    ].copy()

    df["label"] = (
        df["label"]
        .astype(int)
    )

    print(
        f"\nUsable rows after validation: "
        f"{len(df)}"
    )


    # ========================================================
    # 4. LABEL DISTRIBUTION
    # ========================================================

    print("\nLabel distribution:")

    label_counts = (
        df["label"]
        .value_counts()
        .sort_index()
    )

    print(label_counts)

    print("\nLabel meaning:")
    print("0 = Ham")
    print("1 = Spam")


    # ========================================================
    # 5. FEATURES AND TARGET
    # ========================================================

    # X contains cleaned email text
    X = df["clean_text"]

    # y contains spam/ham labels
    y = df["label"]


    # ========================================================
    # 6. TRAIN-TEST SPLIT
    # ========================================================

    print("\n" + "=" * 70)
    print("TRAIN-TEST SPLIT")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(
        f"\nTotal samples   : {len(df)}"
    )

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples : {len(X_test)}"
    )

    print("\nTraining label distribution:")
    print(
        y_train
        .value_counts()
        .sort_index()
    )

    print("\nTesting label distribution:")
    print(
        y_test
        .value_counts()
        .sort_index()
    )


    results = []


    # ========================================================
    # 7. BASELINE NAIVE BAYES
    # ========================================================

    print("\n" + "=" * 70)
    print("BASELINE NAIVE BAYES")
    print("=" * 70)

    naive_bayes = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(),
            ),
            (
                "classifier",
                MultinomialNB(),
            ),
        ]
    )

    print("\nTraining Naive Bayes...")

    naive_bayes.fit(
        X_train,
        y_train,
    )

    nb_feature_count = len(
        naive_bayes
        .named_steps["tfidf"]
        .get_feature_names_out()
    )

    print(
        f"TF-IDF features generated: "
        f"{nb_feature_count}"
    )

    nb_baseline_result = evaluate_model(
        model_name="Naive Bayes",
        stage="Baseline",
        model=naive_bayes,
        X_test=X_test,
        y_test=y_test,
    )

    results.append(
        nb_baseline_result
    )


    # ========================================================
    # 8. BASELINE LOGISTIC REGRESSION
    # ========================================================

    print("\n" + "=" * 70)
    print("BASELINE LOGISTIC REGRESSION")
    print("=" * 70)

    logistic_regression = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    print(
        "\nTraining Logistic Regression..."
    )

    logistic_regression.fit(
        X_train,
        y_train,
    )

    lr_feature_count = len(
        logistic_regression
        .named_steps["tfidf"]
        .get_feature_names_out()
    )

    print(
        f"TF-IDF features generated: "
        f"{lr_feature_count}"
    )

    lr_baseline_result = evaluate_model(
        model_name="Logistic Regression",
        stage="Baseline",
        model=logistic_regression,
        X_test=X_test,
        y_test=y_test,
    )

    results.append(
        lr_baseline_result
    )


    # ========================================================
    # 9. BASELINE MODEL COMPARISON
    # ========================================================

    print("\n" + "=" * 70)
    print("BASELINE MODEL COMPARISON")
    print("=" * 70)

    baseline_df = pd.DataFrame(
        [
            nb_baseline_result,
            lr_baseline_result,
        ]
    )

    print(
        baseline_df[
            [
                "Model",
                "Accuracy",
                "Precision",
                "Recall",
                "F1-score",
            ]
        ].to_string(index=False)
    )


    # ========================================================
    # 10. NAIVE BAYES HYPERPARAMETER TUNING
    # ========================================================

    print("\n" + "=" * 70)
    print("NAIVE BAYES HYPERPARAMETER TUNING")
    print("=" * 70)

    nb_tuning_pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(),
            ),
            (
                "classifier",
                MultinomialNB(),
            ),
        ]
    )

    nb_parameters = {
        "tfidf__ngram_range": [
            (1, 1),
            (1, 2),
        ],

        "tfidf__min_df": [
            1,
            2,
        ],

        "classifier__alpha": [
            0.1,
            0.5,
            1.0,
        ],
    }

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    nb_grid = GridSearchCV(
        estimator=nb_tuning_pipeline,
        param_grid=nb_parameters,
        scoring=scoring,
        refit="accuracy",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )

    nb_grid.fit(
        X_train,
        y_train,
    )

    print(
        "\nBest Naive Bayes parameters:"
    )

    print(
        nb_grid.best_params_
    )

    print(
        "Best Naive Bayes "
        f"cross-validation accuracy: "
        f"{nb_grid.best_score_:.4f}"
    )

    best_nb = (
        nb_grid.best_estimator_
    )

    nb_tuned_result = evaluate_model(
        model_name="Naive Bayes",
        stage="Tuned",
        model=best_nb,
        X_test=X_test,
        y_test=y_test,
    )

    nb_tuned_result[
        "CV Best Accuracy"
    ] = nb_grid.best_score_

    nb_tuned_result[
        "Best Parameters"
    ] = str(
        nb_grid.best_params_
    )

    results.append(
        nb_tuned_result
    )


    # ========================================================
    # 11. LOGISTIC REGRESSION HYPERPARAMETER TUNING
    # ========================================================

    print("\n" + "=" * 70)
    print("LOGISTIC REGRESSION HYPERPARAMETER TUNING")
    print("=" * 70)

    lr_tuning_pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    lr_parameters = {
        "tfidf__ngram_range": [
            (1, 1),
            (1, 2),
        ],

        "tfidf__min_df": [
            1,
            2,
        ],

        "classifier__C": [
            0.5,
            1.0,
            2.0,
        ],

        "classifier__class_weight": [
            None,
            "balanced",
        ],
    }

    lr_grid = GridSearchCV(
        estimator=lr_tuning_pipeline,
        param_grid=lr_parameters,
        scoring=scoring,
        refit="accuracy",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )

    lr_grid.fit(
        X_train,
        y_train,
    )

    print(
        "\nBest Logistic Regression parameters:"
    )

    print(
        lr_grid.best_params_
    )

    print(
        "Best Logistic Regression "
        f"cross-validation accuracy: "
        f"{lr_grid.best_score_:.4f}"
    )

    best_lr = (
        lr_grid.best_estimator_
    )

    lr_tuned_result = evaluate_model(
        model_name="Logistic Regression",
        stage="Tuned",
        model=best_lr,
        X_test=X_test,
        y_test=y_test,
    )

    lr_tuned_result[
        "CV Best Accuracy"
    ] = lr_grid.best_score_

    lr_tuned_result[
        "Best Parameters"
    ] = str(
        lr_grid.best_params_
    )

    results.append(
        lr_tuned_result
    )


    # ========================================================
    # 12. FINAL MODEL COMPARISON
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    results_df = pd.DataFrame(
        results
    )

    display_columns = [
        "Model",
        "Stage",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False
        )
    )


    # ========================================================
    # 13. SAVE MODEL COMPARISON RESULTS
    # ========================================================

    comparison_file = (
        RESULTS_DIR
        / "model_comparison.csv"
    )

    results_df.to_csv(
        comparison_file,
        index=False,
    )

    print(
        f"\nModel comparison saved to:\n"
        f"{comparison_file}"
    )


    # ========================================================
    # 14. SELECT BEST FINAL MODEL
    # ========================================================


    if (
        lr_grid.best_score_
        >= nb_grid.best_score_
    ):

        final_model = best_lr

        final_model_name = (
            "Logistic Regression"
        )

        final_cv_accuracy = (
            lr_grid.best_score_
        )

        final_parameters = (
            lr_grid.best_params_
        )

    else:

        final_model = best_nb

        final_model_name = (
            "Naive Bayes"
        )

        final_cv_accuracy = (
            nb_grid.best_score_
        )

        final_parameters = (
            nb_grid.best_params_
        )


    print("\n" + "=" * 70)
    print("FINAL MODEL SELECTED")
    print("=" * 70)

    print(
        f"\nSelected model: "
        f"{final_model_name}"
    )

    print(
        f"Best CV accuracy: "
        f"{final_cv_accuracy:.4f}"
    )

    print(
        f"Best parameters: "
        f"{final_parameters}"
    )


    # ========================================================
    # 15. FINAL TEST-SET EVALUATION
    # ========================================================

    print("\nFinal held-out test-set performance:")

    final_test_result = evaluate_model(
        model_name=final_model_name,
        stage="Final",
        model=final_model,
        X_test=X_test,
        y_test=y_test,
    )


    # ========================================================
    # 16. SAVE FINAL MODEL
    # ========================================================

    model_file = (
        MODEL_DIR
        / "spam_classifier.joblib"
    )

    joblib.dump(
        final_model,
        model_file,
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"\nFinal model: "
        f"{final_model_name}"
    )

    print(
        f"Final test accuracy: "
        f"{final_test_result['Accuracy']:.4f}"
    )

    print(
        f"Final test precision: "
        f"{final_test_result['Precision']:.4f}"
    )

    print(
        f"Final test recall: "
        f"{final_test_result['Recall']:.4f}"
    )

    print(
        f"Final test F1-score: "
        f"{final_test_result['F1-score']:.4f}"
    )

    print(
        f"\nSaved model:\n"
        f"{model_file}"
    )

    print(
        f"\nSaved evaluation results:\n"
        f"{RESULTS_DIR}"
    )


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    main()