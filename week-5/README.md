# Week 5: Task 3 - Housing Price Prediction (Part 1)

Arch Technologies Internship — Machine Learning Domain (Month 2)

This repository contains the Week 5 deliverables for **Task 3: Housing Price Prediction** using the California Housing dataset. This phase focuses on the first half of the task: Exploratory Data Analysis, Data Cleaning, Feature Engineering, Feature Selection & Preprocessing, and a Baseline Linear Regression model.

---

## Weekly Progression & Deliverables

1. **Monday — Data Loading & Exploration (`notebooks/housing_eda.ipynb`)**
   - Data loading from `data/california-dataset.csv`.
   - Dimensions, dtypes, `.info()`, and `.describe()`.
   - Missing values identification (207 entries in `total_bedrooms`).
   - Distribution histograms and geographic price visualizations.

2. **Tuesday — Data Cleaning (`notebooks/data_cleaning.ipynb` & `outputs/housing_cleaned.csv`)**
   - Missing value imputation in `total_bedrooms` using median (435.0).
   - One-hot encoding of categorical feature `ocean_proximity`.
   - Detection and filtering of extreme recording anomalies (`total_rooms > 25000`, `population > 15000`, `households > 5000`).
   - Cleaned dataset export to `outputs/housing_cleaned.csv`.

3. **Wednesday — Feature Engineering (`notebooks/feature_engineering.ipynb` & `outputs/correlation_heatmap.png`)**
   - Created interaction features:
     - `rooms_per_household = total_rooms / households`
     - `bedrooms_per_room = total_bedrooms / total_rooms`
     - `population_per_household = population / households`
   - Generated full correlation matrix and exported high-resolution heatmap to `outputs/correlation_heatmap.png`.

4. **Thursday — Feature Selection & Preprocessing (`outputs/X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`)**
   - Feature selection based on correlation ranking and spatial domain importance.
   - 80/20 train/test split (`random_state=42`).
   - Standardization of numerical features using `StandardScaler` (fit on train, transform on test).
   - Exported preprocessed CSV datasets to `outputs/`.

5. **Friday — Baseline Model (`scripts/train_baseline_model.py` & `outputs/baseline_results.txt`)**
   - Implemented standalone Python training script for Scikit-Learn `LinearRegression`.
   - Evaluated on test set: RMSE, MAE, and R².
   - Exported summary metrics to `outputs/baseline_results.txt`.

---

## Directory Structure

```text
week-5/
├── data/
│   └── california-dataset.csv
├── notebooks/
│   ├── housing_eda.ipynb
│   ├── data_cleaning.ipynb
│   └── feature_engineering.ipynb
├── outputs/
│   ├── housing_cleaned.csv
│   ├── correlation_heatmap.png
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   ├── y_test.csv
│   └── baseline_results.txt
├── scripts/
│   └── train_baseline_model.py
├── README.md
└── requirements.txt
```

---

## Running the Baseline Model

To run the standalone baseline training script:

```bash
python week-5/scripts/train_baseline_model.py
```

Results are displayed in the terminal and saved to `week-5/outputs/baseline_results.txt`.

---

## Baseline Performance & Week 6 Outlook

- **Test RMSE**: $71,750.18
- **Test MAE**: $52,006.98
- **Test R² Score**: 0.6274 (62.7% variance explained)

The baseline model performs consistently between training (R² = 0.6387) and testing (R² = 0.6274), showing no significant overfitting. However, linear regression is unable to model complex geographic boundary effects and interaction non-linearities. In **Week 6**, advanced ensemble models (Random Forest and Gradient Boosting / XGBoost) along with hyperparameter optimization will be applied to substantially elevate prediction accuracy.
