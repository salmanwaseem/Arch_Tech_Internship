# Week 8: Task 4 - Iris Flower Classification (Part 2) & Month 2 Finalization

Arch Technologies Internship — Machine Learning Domain (Month 2)  
**Author:** Sharjeel Shahzad

This directory contains the Week 8 deliverables for **Task 4: Iris Flower Classification** using the Iris dataset, covering the second half of the task: advanced classifier training, comprehensive multi-metric evaluation, cross-validated hyperparameter tuning, final sample testing, and compiling the official Month 2 Project Report (Task 3 + Task 4).

---

## Weekly Progression & Deliverables

1. **Monday — Advanced Model Training (`scripts/train_advanced_classifiers.py` & `outputs/`)**
   - Loaded preprocessed partitions from `week-7/outputs/` (`X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`).
   - Standardized features with `StandardScaler` fitted on training features.
   - Trained three advanced classification architectures using Scikit-Learn:
     - K-Nearest Neighbors (`KNeighborsClassifier(n_neighbors=5)`)
     - Decision Tree (`DecisionTreeClassifier(criterion='gini', random_state=42)`)
     - Support Vector Machine (`SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)`)
   - Serialized models to `outputs/knn_model.pkl`, `outputs/decision_tree_model.pkl`, and `outputs/svm_model.pkl`.

2. **Tuesday — Model Evaluation & Comparison (`notebooks/iris_evaluation.ipynb` & `outputs/model_comparison_chart.png`)**
   - Evaluated all three models alongside the Week 7 Logistic Regression baseline on the 30-instance test set.
   - Evaluated Accuracy, Macro Precision, Macro Recall, and Macro F1-Score.
   - Rendered publication-quality 4-metric comparative bar chart: `outputs/model_comparison_chart.png`.
   - Documented interactive evaluation workflows in `notebooks/iris_evaluation.ipynb`.

3. **Wednesday — Hyperparameter Tuning (`outputs/hyperparameter_tuning_notes.md` & `outputs/iris_svm_model_tuned.pkl`)**
   - Selected the top-performing model (Support Vector Machine) for systematic optimization.
   - Performed exhaustive 5-Fold Stratified `GridSearchCV` across kernel families (linear, rbf, poly), regularization ($C$), and RBF bandwidth ($\gamma$).
   - Optimal parameters: `kernel='rbf'`, `C=2.0`, `gamma=0.05` (Cross-Validation Accuracy: 96.67%, Test Accuracy: 100.00%).
   - Serialized optimal tuned model to `outputs/iris_svm_model_tuned.pkl`.
   - Documented tuning decisions in `outputs/hyperparameter_tuning_notes.md`.

4. **Thursday — Final Model Testing (`outputs/sample_predictions.png` & `outputs/test_results_summary.txt`)**
   - Loaded tuned SVM and executed inference on representative test flower samples across all three species (*Iris-setosa*, *Iris-versicolor*, *Iris-virginica*).
   - Generated composite visual with sample prediction cards and 3x3 confusion matrix: `outputs/sample_predictions.png`.
   - Exported comprehensive classification test report to `outputs/test_results_summary.txt`.

5. **Friday — Finalization & Month 2 Final Report (`outputs/task4_summary.md` & `outputs/Month2_Report.docx`)**
   - Compiled markdown summary `outputs/task4_summary.md` detailing models trained, benchmark metrics, baseline comparison, and class separability resolutions.
   - Compiled official `outputs/Month2_Report.docx` synthesizing both Task 3 (California Housing Price Prediction, Weeks 5-6) and Task 4 (Iris Flower Classification, Weeks 7-8) adhering to Arch Technologies front-page guidelines.

---

---

## Execution Instructions

To execute the complete pipeline and regenerate all models, charts, and metrics:

```bash
# Run standalone end-to-end training and evaluation pipeline
python week-8/scripts/train_advanced_classifiers.py

# Recompile the comprehensive Month 2 Report (.docx)
python week-8/scripts/build_month2_report.py
```

