# Feature Engineering & Data Preprocessing

## Project Overview
This repository demonstrates end-to-end feature engineering and data preprocessing techniques applied to a customer churn dataset, transforming raw records into clean, model-ready features.

## Repository Structure
- `README.md`: Project overview and navigation guide
- `requirements.txt`: Required Python libraries (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`)
- `data/raw_dataset.csv`: Original unscaled customer dataset
- `data/updated_dataset.csv`: Processed, encoded, and standardized model-ready dataset
- `notebooks/feature_engineering_preprocessing.ipynb`: Jupyter Notebook demonstrating feature creation, encoding, scaling, and feature selection
- `docs/preprocessing_documentation.md`: Short documentation explaining features created and why preprocessing was required

## Techniques Implemented
1. **Feature Creation**: Engineered `AvgMonthlySpend`, `HighTicketRiskFlag`, and `SpendPerAgeRatio`.
2. **Encoding Techniques**: Applied Ordinal Encoding (`ContractEncoded`) and One-Hot Encoding (`ContractType_One year`, `ContractType_Two year`).
3. **Scaling Methods**: Compared `MinMaxScaler` (`[0, 1]` range) and applied `StandardScaler` (zero mean, unit variance).
4. **Feature Selection**: Ranked features using `SelectKBest` with ANOVA F-test (`f_classif`).

## How to Run
1. Clone the repository:
   `git clone https://github.com/adhikarysayantani7-tony/feature-engineering-preprocessing.git`
2. Install dependencies:
   `pip install -r requirements.txt`
3. Open the notebook:
   `jupyter notebook notebooks/feature_engineering_preprocessing.ipynb`
