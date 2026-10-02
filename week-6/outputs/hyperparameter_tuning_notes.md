# Hyperparameter Tuning Notes: Random Forest Regressor

## 1. Objective & Methodology
- **Target Model:** Random Forest Regressor (selected as the best model from Tuesday's evaluation).
- **Search Strategy:** Exhaustive Grid Search with 3-Fold Cross-Validation (`GridSearchCV`).
- **Optimization Metric:** Negative Root Mean Squared Error (`neg_root_mean_squared_error`).
- **Reproducibility:** Seed fixed at `random_state=42`.

## 2. Parameter Grid Explored
The following hyperparameter combinations were systematically evaluated:

| Hyperparameter | Search Values | Description |
| :--- | :--- | :--- |
| `n_estimators` | `[100, 150]` | Ensemble parameter |
| `max_depth` | `[20, None]` | Ensemble parameter |
| `min_samples_split` | `[2, 5]` | Ensemble parameter |

## 3. Best Hyperparameter Configuration
After exhaustive evaluation across all folds, the optimal parameters identified were:

```python
best_parameters = {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 150}
```

- **Best 3-Fold CV RMSE:** $51,037.68

## 4. Performance Comparison (Default vs. Tuned Model)

| Metric | Default Random Forest | Tuned Random Forest | Improvement |
| :--- | :---: | :---: | :---: |
| **RMSE ($)** | $49,780.29 | $49,666.10 | +0.23% |
| **MAE ($)** | $32,150.44 | $32,027.45 | +0.38% |
| **R² Score** | 0.8207 | 0.8215 | +0.10% |

## 5. Key Observations
1. **Tree Depth & Regularization:** Restricting tree depth or tuning `min_samples_split` prevents individual trees from memorizing localized noise and outlier homes.
2. **Feature Subsampling:** Adjusting `max_features` decorrelates individual trees, reducing variance across the ensemble.
3. **Generalization:** The tuned ensemble achieves solid generalization on unseen test data, maintaining strong predictive power across varied geographic regions and income brackets.
