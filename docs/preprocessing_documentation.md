# Feature Engineering & Data Preprocessing Documentation

## 1. Overview
Raw customer data cannot be fed directly into most machine learning algorithms due to text categories, wide numerical scale differences, and hidden variable interactions. This document explains the features engineered and why each preprocessing step was required.

---

## 2. Features Created
1. **`AvgMonthlySpend` (`TotalCharges / TenureMonths`)**:
   * *Purpose*: Captures a customer's true average monthly spend over their lifetime, smoothing out promotional or one-time billing spikes.
2. **`HighTicketRiskFlag` (`1` if `SupportTickets >= 3`, else `0`)**:
   * *Purpose*: Converts raw support ticket counts into a clear binary risk indicator, separating satisfied customers from frustrated users likely to churn.
3. **`SpendPerAgeRatio` (`MonthlyCharges / Age`)**:
   * *Purpose*: Measures monthly financial commitment relative to customer age, highlighting younger users on high-cost plans who are more price-sensitive.

---

## 3. Why Preprocessing Was Required

### A. Categorical Encoding (`ContractType`)
* **Why Required**: Machine learning algorithms operate on numbers, not text strings like `"Month-to-month"` or `"Two year"`.
* **Techniques Applied**:
  * **Ordinal Encoding (`ContractEncoded`)**: Preserved the natural duration order (`0`, `1`, `2`).
  * **One-Hot Encoding (`ContractType_One year`, `ContractType_Two year`)**: Created binary flags with `drop_first=True` to avoid multicollinearity (the dummy variable trap).

### B. Numerical Scaling (`StandardScaler` & `MinMaxScaler`)
* **Why Required**: Raw features span vastly different ranges (`TotalCharges` reaches `5730.00`, while `SupportTickets` ranges from `0` to `6`). Without scaling, distance-based and gradient-based models (like Logistic Regression, KNN, and SVM) become biased toward large-magnitude numbers.
* **Techniques Applied**:
  * **`MinMaxScaler`**: Tested for bounded `[0, 1]` normalization.
  * **`StandardScaler`**: Applied to center features at mean `0` and standard deviation `1` for the final updated dataset.

### C. Feature Selection (`SelectKBest` with ANOVA F-Test)
* **Why Required**: Removing redundant or weak features reduces noise, speeds up training, and prevents overfitting.
* **Outcome**: Ranked features statistically and identified `HighTicketRiskFlag`, `ContractEncoded`, `SupportTickets_Scaled`, `TenureMonths_Scaled`, and `TotalCharges_Scaled` as the strongest predictors of customer churn.
