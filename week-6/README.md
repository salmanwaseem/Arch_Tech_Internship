# Week 6: Housing Price Prediction (Advanced Models, Tuning & Final Evaluation)

Week 6 concludes **Task 3: Housing Price Prediction** on the California Housing dataset (Month 2). Building upon the preprocessed data and Linear Regression baseline established in Week 5, this week focuses on advanced non-linear regression ensembles (Random Forest Regressor and Gradient Boosting Regressor), comparative evaluation, hyperparameter optimization via `GridSearchCV`, held-out sample testing, and complete project documentation.

---

## Deliverables Breakdown

- **Monday (Advanced Model Training):**
  - Script: `scripts/train_advanced_models.py`
  - Models: `outputs/random_forest_model.pkl` and `outputs/gradient_boosting_model.pkl`
- **Tuesday (Model Evaluation & Comparison):**
  - Evaluation on test set: RMSE, MAE, R²
  - Baseline comparison against Week 5 Linear Regression (`week-5/outputs/baseline_results.txt`)
  - Comparative visualization: `outputs/model_comparison_chart.png`
  - Interactive analysis notebook: `notebooks/housing_evaluation.ipynb`
- **Wednesday (Hyperparameter Tuning):**
  - Cross-validated grid search (`GridSearchCV`) on Random Forest
  - Tuning notes & comparison: `outputs/hyperparameter_tuning_notes.md`
  - Final tuned model: `outputs/housing_rf_model_tuned.pkl`
- **Thursday (Final Model Testing & Diagnostics):**
  - Sample predictions on held-out test instances
  - Actual vs. predicted scatter plot: `outputs/sample_predictions.png`
  - Metrics and sample error table: `outputs/test_results_summary.txt`
- **Friday (Finalization & Documentation):**
  - Comprehensive summary: `outputs/task3_summary.md`

---

## Directory Structure

```text
week-6/
├── data/
│   └── california-dataset.csv          # Reference copy of raw dataset
├── notebooks/
│   └── housing_evaluation.ipynb        # Interactive evaluation & visualization notebook
├── outputs/
│   ├── gradient_boosting_model.pkl     # Trained Gradient Boosting model
│   ├── housing_rf_model_tuned.pkl      # Final tuned Random Forest model
│   ├── hyperparameter_tuning_notes.md  # GridSearchCV documentation
│   ├── model_comparison_chart.png      # Multi-metric comparative bar chart
│   ├── random_forest_model.pkl         # Trained baseline Random Forest model
│   ├── sample_predictions.png          # Predicted vs Actual scatter plot
│   ├── task3_summary.md                # Final Task 3 markdown report
│   └── test_results_summary.txt        # Numerical test results and sample table
├── scripts/
│   └── train_advanced_models.py        # End-to-end training and evaluation script
├── requirements.txt                    # Project Python dependencies
└── README.md                           # Week 6 overview and instructions
```

---

## Quick Start & Reproducibility

From the repository root:

```bash
# 1. Install dependencies
pip install -r week-6/requirements.txt

# 2. Run standalone training, tuning, and evaluation pipeline
python week-6/scripts/train_advanced_models.py

# 3. Launch interactive evaluation notebook
jupyter notebook week-6/notebooks/housing_evaluation.ipynb
```

---

## Model Performance Summary

| Model | RMSE ($) | MAE ($) | R² Score |
| :--- | :---: | :---: | :---: |
| **Linear Regression (Baseline)** | $69,127.04 | $49,645.49 | 0.6353 |
| **Gradient Boosting Regressor** | $53,505.02 | $36,404.13 | 0.7815 |
| **Random Forest Regressor (Default)** | $49,833.73 | $31,926.27 | 0.8105 |
| **Random Forest Regressor (Tuned)** | **$49,638.12** | **$31,870.45** | **0.8120** |

The final tuned Random Forest ensemble achieves a **28.2% reduction in RMSE** and a **35.8% reduction in MAE** compared to the Week 5 baseline model.

---

## Next Steps

Task 3 is complete. **Task 4: Iris Flower Classification** begins in Week 7.

