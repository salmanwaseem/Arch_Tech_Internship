# Week 2 - Email Spam Classification

## Objective

The objective of Week 2 was to build and evaluate a machine learning
model for classifying emails as Spam or Ham using the cleaned dataset
prepared during Week 1.

## Work Completed

### 1. Feature Extraction

Implemented text feature extraction in:

`src/features.py`

The following techniques were explored:

- Bag-of-Words using `CountVectorizer`
- TF-IDF using `TfidfVectorizer`

The cleaned email text from Week 1 was converted into numerical feature
vectors that can be used by machine learning algorithms.

### 2. Train-Test Split

The cleaned dataset was divided into training and testing sets using
Scikit-learn.

- Training samples: 65,319
- Testing samples: 16,330

### 3. Model Training

Two machine learning models were trained:

- Multinomial Naive Bayes
- Logistic Regression

Training was implemented in:

`src/train.py`

### 4. Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The evaluation results are available in:

`results/`

A separate evaluation notebook was also created:

`notebooks/model_evaluation.ipynb`

### 5. Hyperparameter Tuning

`GridSearchCV` was used to tune both Naive Bayes and Logistic Regression.

The tuned Logistic Regression model achieved the best overall performance.

### 6. Final Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 98.65% |
| Precision | 98.34% |
| Recall | 99.14% |
| F1-score | 98.74% |

The final selected model was:

**Logistic Regression**

The trained model is saved in:

`models/spam_classifier.joblib`

### 7. Sample Email Prediction

A prediction script was created:

`src/predict.py`

The script loads the saved model and tests it on unseen sample emails.

Example output:

- Spam promotional email → `SPAM`
- Normal meeting/project email → `HAM`

### Project Structure

```text
week-2/
│
├── src/
│   ├── features.py
│   ├── train.py
│   └── predict.py
│
├── notebooks/
│   └── model_evaluation.ipynb
│
├── models/
│   └── spam_classifier.joblib
│
├── results/
│   ├── model_comparison.csv
│   ├── naive_bayes_baseline_confusion_matrix.png
│   ├── naive_bayes_tuned_confusion_matrix.png
│   ├── logistic_regression_baseline_confusion_matrix.png
│   ├── logistic_regression_tuned_confusion_matrix.png
│   └── logistic_regression_final_confusion_matrix.png
│
├── README.md
└── requirements.txt

Conclusion

Both Naive Bayes and Logistic Regression performed well on the spam
classification task. After hyperparameter tuning, Logistic Regression
was selected as the final model because it achieved the best overall
performance on the held-out test dataset.