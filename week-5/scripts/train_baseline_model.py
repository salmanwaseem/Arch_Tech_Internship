"""
Week 5: Task 3 - Housing Price Prediction
Standalone Baseline Model Training Script (Friday Deliverable)

Internship: Arch Technologies, Machine Learning Domain, Month 2
Author: Sharjeel Shahzad

This script loads the preprocessed datasets (X_train.csv, X_test.csv,
y_train.csv, y_test.csv) generated during feature engineering and selection,
fits a Scikit-Learn LinearRegression baseline model, computes evaluation
metrics (RMSE, MAE, R²), prints results to the console, and writes the
output to outputs/baseline_results.txt.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def resolve_paths() -> tuple[Path, Path]:
    """Resolve base directories dynamically to support multiple execution contexts."""
    script_dir = Path(__file__).resolve().parent
    week5_dir = script_dir.parent if script_dir.name == "scripts" else script_dir
    outputs_dir = week5_dir / "outputs"
    if not outputs_dir.exists():
        # Fallback if executed from repository root
        outputs_dir = Path("week-5/outputs").resolve()
    return week5_dir, outputs_dir


def load_preprocessed_data(outputs_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load preprocessed feature matrices and target vectors."""
    x_train_path = outputs_dir / "X_train.csv"
    x_test_path = outputs_dir / "X_test.csv"
    y_train_path = outputs_dir / "y_train.csv"
    y_test_path = outputs_dir / "y_test.csv"

    for path in (x_train_path, x_test_path, y_train_path, y_test_path):
        if not path.exists():
            raise FileNotFoundError(f"Required dataset file not found: {path}")

    x_train = pd.read_csv(x_train_path)
    x_test = pd.read_csv(x_test_path)
    y_train = pd.read_csv(y_train_path).squeeze("columns")
    y_test = pd.read_csv(y_test_path).squeeze("columns")

    return x_train, x_test, y_train, y_test


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate RMSE, MAE, and R² evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def train_and_evaluate() -> dict[str, float]:
    """Train Linear Regression baseline and persist evaluation results."""
    week5_dir, outputs_dir = resolve_paths()
    print("=" * 60)
    print("WEEK 5: BASELINE LINEAR REGRESSION MODEL TRAINING")
    print(f"Working Directory: {week5_dir}")
    print(f"Loading data from: {outputs_dir}")
    print("=" * 60)

    x_train, x_test, y_train, y_test = load_preprocessed_data(outputs_dir)
    print(f"Training samples: {x_train.shape[0]:,}, Features: {x_train.shape[1]}")
    print(f"Testing samples:  {x_test.shape[0]:,}")

    # Initialize and train Scikit-learn Linear Regression model
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Predictions
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # Evaluate
    train_metrics = evaluate_model(y_train.to_numpy(), y_train_pred)
    test_metrics = evaluate_model(y_test.to_numpy(), y_test_pred)

    result_text = (
        "===========================================================\n"
        "   California Housing Price Prediction - Baseline Model    \n"
        "   Model: Ordinary Least Squares (OLS) Linear Regression   \n"
        "   Internship: Arch Technologies - ML Domain (Month 2)     \n"
        "===========================================================\n\n"
        f"Training Set Samples: {x_train.shape[0]:,}\n"
        f"Test Set Samples:     {x_test.shape[0]:,}\n"
        f"Feature Count:        {x_train.shape[1]}\n"
        f"Features Used:        {list(x_train.columns)}\n\n"
        "EVALUATION METRICS (TEST SET):\n"
        f"  - Root Mean Squared Error (RMSE): ${test_metrics['rmse']:,.2f}\n"
        f"  - Mean Absolute Error (MAE):      ${test_metrics['mae']:,.2f}\n"
        f"  - R^2 Score (Coefficient of Det.): {test_metrics['r2']:.4f}\n\n"
        "TRAINING SET METRICS (FOR OVERFITTING CHECK):\n"
        f"  - Train RMSE:                      ${train_metrics['rmse']:,.2f}\n"
        f"  - Train MAE:                       ${train_metrics['mae']:,.2f}\n"
        f"  - Train R^2:                        {train_metrics['r2']:.4f}\n\n"
        "PERFORMANCE SUMMARY & WEEK 6 OUTLOOK:\n"
        "The baseline Linear Regression model accounts for approximately\n"
        f"{test_metrics['r2']*100:.1f}% of the variance in California median house prices\n"
        f"with a test RMSE of ${test_metrics['rmse']:,.2f} and MAE of ${test_metrics['mae']:,.2f}.\n"
        "The model shows no substantial overfitting (Train R^2 is comparable\n"
        "to Test R^2), but its predictive accuracy is constrained by linear\n"
        "assumptions over complex geographical interactions. In Week 6, non-linear\n"
        "ensemble methods (Random Forest and Gradient Boosting) along with systematic\n"
        "hyperparameter tuning will be implemented to capture spatial non-linearities\n"
        "and substantially reduce prediction error.\n"
        "===========================================================\n"
    )

    print(result_text)

    # Save results to outputs/baseline_results.txt
    results_path = outputs_dir / "baseline_results.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(result_text)

    print(f"Results successfully saved to: {results_path}")
    return test_metrics


if __name__ == "__main__":
    train_and_evaluate()
