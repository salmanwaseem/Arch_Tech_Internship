# Task 3 Final Summary: Housing Price Prediction

**Internship Domain:** Machine Learning  
**Project:** California Housing Price Prediction (Month 2, Weeks 5 & 6)  
**Deliverable Status:** Complete  

---

## 1. Models Trained & Final Model Selection
During Task 3, three distinct machine learning paradigms were trained and systematically benchmarked on the California Housing dataset:
1. **Baseline Model (Week 5):** Multiple Linear Regression (ordinary least squares on scaled features).
2. **Advanced Ensemble 1 (Week 6):** Random Forest Regressor (`n_estimators=100`, bagging ensemble).
3. **Advanced Ensemble 2 (Week 6):** Gradient Boosting Regressor (`n_estimators=100`, boosting ensemble).
4. **Final Model:** **Tuned Random Forest Regressor** optimized via `GridSearchCV`.

### Why Random Forest Was Chosen
- **Non-Linear Relationships:** Housing prices exhibit pronounced non-linear dependencies on geographic coordinates (`latitude`, `longitude`) and economic indicators (`median_income`), which tree ensembles capture naturally.
- **Superior Variance Reduction:** Bagging across randomized feature subsets proved more resilient to localized dataset outliers than linear formulations.
- **Top Empirical Score:** Random Forest outperformed Gradient Boosting across all key evaluation criteria (RMSE: $49,666.10 vs. $57,393.75; R²: 0.8215 vs. 0.7616).

---

## 2. Final Evaluation Metrics
On the held-out test set (4,128 unseen housing block instances), the final tuned model achieved:
- **Root Mean Squared Error (RMSE):** **$49,666.10**
- **Mean Absolute Error (MAE):** **$32,027.45**
- **R-squared (R²) Score:** **0.8215** (explaining over 82% of house value variance)

---

## 3. Comparison Against Week 5 Baseline

| Metric | Week 5 Baseline (Linear Regression) | Final Model (Tuned Random Forest) | Net Improvement |
| :--- | :---: | :---: | :---: |
| **RMSE ($)** | $71,750.18 | $49,666.10 | **30.78% reduction** |
| **MAE ($)** | $52,006.98 | $32,027.45 | **38.42% reduction** |
| **R² Score** | 0.6353 | 0.8215 | **+29.31% gain** |

The transition from a linear baseline to an advanced non-linear ensemble yielded a massive error reduction, lowering average prediction error by over $19,979.53 per house.

---

## 4. Key Challenges & Resolutions
1. **Feature Engineering & Multicollinearity:**
   - *Challenge:* Raw counts (`total_rooms`, `total_bedrooms`, `population`) were heavily correlated with cluster population rather than living standards.
   - *Resolution:* Engineered ratio features (`rooms_per_household`, `bedrooms_per_room`, `population_per_household`) during Week 5 provided interpretable, per-household density signals that gave high feature importance in tree splits.
2. **Artificial Target Capping at $500,001:**
   - *Challenge:* In the California census survey, values above $500,000 were truncated to $500,001, introducing a horizontal ceiling artifact that penalizes standard regression models.
   - *Resolution:* Random Forest handles ceiling discontinuities substantially better than linear regression, as decision boundaries split at threshold values without distorting predictions for lower price ranges.
3. **Overfitting Mitigation in Deep Trees:**
   - *Challenge:* Unconstrained decision trees tend to memorize training outliers, resulting in overly optimistic training scores but degraded test generalization.
   - *Resolution:* Tuned `max_depth`, `min_samples_split`, and `max_features` through cross-validation to restrict leaf node granularity and decorrelate individual trees.

---

## 5. Transition to Task 4
With Task 3 fully trained, evaluated, and documented, all deliverables for Housing Price Prediction are concluded. **Task 4: Iris Flower Classification** begins next in Week 7.
