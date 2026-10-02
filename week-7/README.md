# Week 7: Task 4 - Iris Flower Classification (Part 1)

Arch Technologies Internship — Machine Learning Domain (Month 2)

This repository contains the Week 7 deliverables for **Task 4: Iris Flower Classification** using the Kaggle UCI Iris dataset (`iris-dataset.csv`). This phase implements the first half of the task: Exploratory Data Analysis, Multi-Feature Visualization, Data Preprocessing & Outlier Evaluation, Feature Scaling, Baseline Logistic Regression Classification, and Model Evaluation.

---

## Weekly Progression & Deliverables

1. **Monday — Data Loading & Exploration (`notebooks/iris_eda.ipynb`)**
   - Loaded `data/iris-dataset.csv` into a Pandas DataFrame.
   - Dropped non-informative `Id` row identifier.
   - Evaluated dimensions (150 rows, 5 columns), `.dtypes`, `.info()`, and `.describe()`.
   - Verified 0 missing values and verified perfectly balanced class distribution (50 samples each for Setosa, Versicolor, and Virginica).

2. **Tuesday — Exploratory Data Visualization (`notebooks/iris_visualization.ipynb` & `outputs/`)**
   - Constructed comprehensive pair plot (`outputs/pairplot.png`) showing pairwise interactions and KDE distributions across all four morphological features.
   - Built dual-panel scatter plot (`outputs/scatterplot.png`) evaluating class separability:
     - Petal Length vs. Petal Width (demonstrating clear linear separability of Setosa).
     - Sepal Length vs. Sepal Width (capturing morphological spread).
   - Computed Pearson correlation matrix ($r = 0.963$ between petal length and petal width).

3. **Wednesday — Preprocessing (`notebooks/preprocessing.ipynb` & `outputs/`)**
   - Transformed categorical `Species` into numeric labels via `LabelEncoder` (`0: Iris-setosa`, `1: Iris-versicolor`, `2: Iris-virginica`).
   - Conducted Interquartile Range (IQR) outlier detection across all 4 features:
     - Identified 4 statistical candidate points in `SepalWidthCm`.
     - Analyzed biological significance and preserved them as valid morphological variance.
   - Split dataset into 80/20 train/test partitions (`random_state=42`).
   - Exported preprocessed CSV datasets to `outputs/X_train.csv`, `outputs/X_test.csv`, `outputs/y_train.csv`, `outputs/y_test.csv`.

4. **Thursday — Feature Scaling & Baseline Model (`scripts/train_baseline_classifier.py` & `outputs/baseline_logistic_model.pkl`)**
   - Scaled continuous features using `StandardScaler` (fit on train, transformed on test).
   - Fitted Scikit-Learn multinomial `LogisticRegression(max_iter=200, random_state=42)`.
   - Serialized and saved trained baseline classifier to `outputs/baseline_logistic_model.pkl`.

5. **Friday — Baseline Evaluation (`outputs/confusion_matrix.png` & `outputs/baseline_results.txt`)**
   - Evaluated accuracy, 3x3 confusion matrix, and full classification report.
   - Generated and exported confusion matrix heatmap to `outputs/confusion_matrix.png`.
   - Compiled detailed metrics and class separability commentary to `outputs/baseline_results.txt`.

---

## Directory Structure

```text
week-7/
├── data/
│   └── iris-dataset.csv
├── notebooks/
│   ├── iris_eda.ipynb
│   ├── iris_visualization.ipynb
│   └── preprocessing.ipynb
├── outputs/
│   ├── pairplot.png
│   ├── scatterplot.png
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   ├── y_test.csv
│   ├── baseline_logistic_model.pkl
│   ├── confusion_matrix.png
│   └── baseline_results.txt
├── scripts/
│   └── train_baseline_classifier.py
├── README.md
└── requirements.txt
```

---

## Running the Baseline Classifier

To run the standalone baseline training and evaluation script:

```bash
python week-7/scripts/train_baseline_classifier.py
```

Results are printed directly to the terminal and automatically persisted to `week-7/outputs/baseline_results.txt`.

---

## Baseline Performance & Week 8 Outlook

- **Training Accuracy**: 96.67% (116 / 120 correct)
- **Testing Accuracy**: 100.00% (30 / 30 correct)
- **Macro F1-Score (Test)**: 1.0000

### Analytical Findings:
1. **Linear Separability**: `Iris-setosa` is completely linearly separable from the other two species across both sepal and petal features, achieving 100% precision and recall with zero ambiguity.
2. **Boundary Overlap**: `Iris-versicolor` and `Iris-virginica` exhibit slight morphological overlap in feature space. While the 30-sample holdout test partition is partitioned cleanly by the linear hyperplane (100% test accuracy), the 120-sample training set has 4 boundary points (3 Versicolor classified as Virginica, 1 Virginica classified as Versicolor; 96.67% train accuracy).
3. **Week 8 Outlook**: In **Week 8**, non-linear classifiers (K-Nearest Neighbors, Decision Trees, and Support Vector Machines with RBF kernel) along with hyperparameter optimization will be implemented to assess non-linear decision boundaries and compare robustness against this Week 7 baseline.
