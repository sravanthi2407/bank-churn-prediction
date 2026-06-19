# 🏦 Predictive Modeling and Risk Scoring for Bank Customer Churn

<p align="center">
  <img src="reports/phase1_data_overview.png" width="700"/>
</p>

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview

A production-ready machine learning system that predicts bank customer churn with **86.8% accuracy** and **0.866 ROC-AUC**, enabling data-driven retention strategies that can reduce churn by an estimated 15–25%.

The project covers the full ML lifecycle — from raw data ingestion through EDA, feature engineering, model training with hyperparameter tuning, explainability analysis, customer risk scoring, and a live Streamlit dashboard.

---

## 🏦 Business Problem

Customer churn costs the banking industry billions annually. Acquiring a new customer costs 5–7× more than retaining an existing one. Identifying at-risk customers **before** they leave allows banks to:

- Deploy targeted retention offers at the right time
- Prioritise retention budget on highest-risk, highest-value customers
- Reduce churn rate and protect net interest income

---

## 📊 Dataset Description

| Attribute         | Details                                      |
|------------------|----------------------------------------------|
| **Source**        | European Bank Customer Records (2025)       |
| **Records**       | 10,000 customers                            |
| **Features**      | 14 columns (11 features + 3 metadata)       |
| **Target**        | `Exited` — Binary (0 = Retained, 1 = Churned) |
| **Class Balance** | 79.6% Retained · 20.4% Churned             |
| **Geographies**   | France (50.1%), Germany (25.1%), Spain (24.8%) |

### Feature Descriptions

| Feature          | Type        | Description                                  |
|-----------------|-------------|----------------------------------------------|
| CreditScore      | Numerical   | Customer credit score (350–850)             |
| Geography        | Categorical | Country: France, Germany, Spain             |
| Gender           | Categorical | Male / Female                               |
| Age              | Numerical   | Customer age (18–92)                        |
| Tenure           | Numerical   | Years as bank customer (0–10)               |
| Balance          | Numerical   | Account balance (€0–250,898)               |
| NumOfProducts    | Numerical   | Number of bank products (1–4)              |
| HasCrCard        | Binary      | 1 = Has credit card                         |
| IsActiveMember   | Binary      | 1 = Active banking member                  |
| EstimatedSalary  | Numerical   | Annual salary estimate                      |
| **Exited**       | **Target**  | **1 = Churned, 0 = Retained**              |

---

## 🏗️ Architecture

```
European_Bank.csv
        │
        ▼
┌──────────────────┐
│  preprocess.py   │  → Drop irrelevant cols → Encode → Feature Engineer → Scale → Split
└──────────────────┘
        │
        ▼
┌──────────────────┐
│  train_model.py  │  → Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
│  (GridSearchCV)  │  → Cross-validation → Best Model Selection
└──────────────────┘
        │
        ▼
┌────────────────────────┐
│  Risk Scoring Engine   │  → ChurnProbability, RiskScore, RiskCategory per customer
└────────────────────────┘
        │
        ▼
┌──────────────┐
│   app.py     │  → Streamlit Dashboard (5 pages)
└──────────────┘
```

---

## 📁 Project Structure

```
Bank_Customer_Churn_Project/
├── data/
│   ├── European_Bank.csv          # Raw dataset
│   └── processed.csv              # Feature-engineered dataset
├── models/
│   ├── best_model.pkl             # Gradient Boosting (best)
│   ├── random_forest.pkl
│   ├── decision_tree.pkl
│   ├── logistic_regression.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   └── feature_cols.pkl
├── notebooks/
│   ├── phase1_data_understanding.py
│   ├── phase2_eda.py
│   ├── phase7_risk_scoring.py
│   └── phase8_explainability.py
├── reports/
│   ├── phase1_data_overview.png
│   ├── phase2_univariate.png
│   ├── phase2_bivariate.png
│   ├── phase2_multivariate.png
│   ├── phase6_evaluation.png
│   ├── phase7_risk_scoring.png
│   ├── phase8_explainability.png
│   ├── phase8_pdp.png
│   ├── model_results.csv
│   ├── feature_importance.csv
│   └── customer_risk_scores.csv
├── app.py                         # Streamlit dashboard
├── preprocess.py                  # Data preprocessing + feature engineering
├── train_model.py                 # Model training + evaluation
├── requirements.txt
└── README.md
```

---

## 🚀 Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/bank-churn-prediction.git
cd bank-churn-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Preprocessing
```bash
python preprocess.py
```

### 4. Train Models
```bash
python train_model.py
```

### 5. Launch Streamlit Dashboard
```bash
streamlit run app.py
```

---

## 📈 Results

| Model               | Accuracy | Precision | Recall | F1     | ROC-AUC |
|--------------------|----------|-----------|--------|--------|---------|
| Logistic Regression | 81.0%   | 63.4%     | 15.7%  | 25.2%  | 0.773   |
| Decision Tree       | 85.5%   | 81.3%     | 37.4%  | 51.2%  | 0.839   |
| Random Forest       | 86.5%   | 80.9%     | 43.7%  | 56.8%  | 0.857   |
| **Gradient Boosting** | **86.8%** | **79.0%** | **47.9%** | **59.6%** | **0.866** |

### Risk Segmentation Results

| Risk Category | Customers | Actual Churn Rate |
|--------------|-----------|-------------------|
| 🟢 Low Risk  | 8,791 (87.9%) | 12.1% |
| 🟡 Medium Risk | 664 (6.6%) | 68.1% |
| 🔴 High Risk | 545 (5.5%) | 95.8% |

### Top Churn Drivers
1. **Age** (39.0% importance) — Customers 40+ churn significantly more
2. **Number of Products** (29.4%) — Over-banked customers (3–4 products) are high risk
3. **Engagement × Products** (9.2%) — Inactive multi-product customers
4. **Balance** (5.1%) — Higher-balance customers more likely to switch
5. **IsActiveMember** (4.1%) — Inactive members churn at 2× the rate

---

## 🧩 Feature Engineering

| Feature                      | Formula                              | Business Meaning                   |
|-----------------------------|--------------------------------------|------------------------------------|
| BalanceSalaryRatio           | Balance / (Salary + 1)              | Relative wealth indicator          |
| ProductDensity               | NumProducts / (Tenure + 1)          | Engagement density per year        |
| EngagementProductInteraction | IsActive × NumProducts              | Loyalty proxy                      |
| AgeTenureInteraction         | Age × Tenure                        | Lifestage-loyalty combination      |
| IsZeroBalance                | Balance == 0                        | Dormant account flag               |
| AgeGroup                     | Age binned 18-30/31-40/41-50/51-60/60+ | Lifecycle segmentation          |
| CreditRisk                   | Inverted CreditScore band           | Credit quality risk tier           |

---

## ☁️ Deployment

### Streamlit Cloud
```bash
# Push to GitHub, then connect at share.streamlit.io
# Set main file: app.py
```

### Render
```bash
# render.yaml
services:
  - type: web
    name: bank-churn-app
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app.py --server.port $PORT
```

### Hugging Face Spaces
```bash
# Create Space → SDK: Streamlit → Upload all files
# Set app_file: app.py in README YAML header
```

---

## 🔮 Future Enhancements

- [ ] XGBoost & LightGBM integration
- [ ] SHAP explainability integration
- [ ] SMOTE for class imbalance handling
- [ ] Real-time prediction API (FastAPI)
- [ ] Customer lifetime value (CLV) integration
- [ ] Time-series churn trend analysis
- [ ] Multi-year dataset support

---

## 👤 Author

Built for academic submission, GitHub portfolio, and LinkedIn showcase.

**Stack:** Python · scikit-learn · pandas · matplotlib · seaborn · Streamlit

---

## 📄 License

MIT License — free for academic and commercial use with attribution.
