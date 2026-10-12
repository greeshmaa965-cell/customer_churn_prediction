# 📊 Explainable Customer Churn Prediction

> **End-to-end machine learning project** for predicting customer churn with explainable AI using Python, Scikit-learn, and SHAP.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-app-red?logo=streamlit)
![SHAP](https://img.shields.io/badge/SHAP-explainability-green)

---

## 🎯 Project Overview

Customer churn — when a customer stops using a service — is a critical problem for telecom companies. This project builds a **full ML pipeline** that:

1. **Analyses** the Telco Customer Churn dataset (IBM)
2. **Engineers features** with OneHotEncoder + StandardScaler
3. **Trains** a Logistic Regression model (threshold tuned to 0.3 for high recall)
4. **Evaluates** with confusion matrix, ROC curve, and threshold analysis
5. **Explains** predictions using SHAP values
6. **Deploys** via an interactive Streamlit dashboard

---

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | **74.95%** |
| Precision | 51.9% |
| Recall | **75.4%** (optimised) |
| F1-Score | 61.5% |
| ROC-AUC | **84.2%** |

> ⚡ The model uses a **0.3 decision threshold** (tuned in notebook 03) to maximise recall — catching as many true churners as possible is more valuable than minimising false positives.

---

## 🗂️ Project Structure

```
customer_churn_prediction/
│
├── data/
│   ├── raw/                        # Original Telco CSV dataset
│   └── processed/                  # Pickled train/test splits
│
├── models/
│   ├── logistic_regression_model.pkl   # Trained model
│   └── preprocessor.pkl                # Fitted ColumnTransformer
│
├── noteboooks/
│   ├── 01_data_understanding.ipynb     # EDA & visualisations
│   ├── 02_feature_engineering.ipynb    # Preprocessing pipeline
│   ├── 03_model_building.ipynb         # Model training & threshold tuning
│   ├── 04_model_evaluation.ipynb       # Full evaluation & error analysis
│   └── 05_SHAP.ipynb                   # SHAP explainability
│
├── reports/
│   └── figures/                    # Saved evaluation plots
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       └── threshold_analysis.png
│
├── src/                            # Reusable Python modules
│   ├── __init__.py
│   ├── data_loader.py              # Load & clean raw CSV
│   ├── preprocessor.py             # ColumnTransformer pipeline
│   ├── train.py                    # Train & save model
│   ├── evaluate.py                 # Generate evaluation plots
│   └── predict.py                  # Single & batch inference
│
├── app/
│   └── streamlit_app.py            # Interactive Streamlit dashboard
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/greeshmaa965-cell/customer_churn_prediction.git
cd customer_churn_prediction
```

### 2. Set up environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Train the model

```bash
python -m src.train
```

### 4. Generate evaluation plots

```bash
python -m src.evaluate
```

### 5. Launch the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

---

## 🔮 Streamlit Dashboard Features

| Page | Description |
|------|-------------|
| **Single Prediction** | Enter customer details and get instant churn probability + SHAP explanation |
| **Batch Prediction** | Upload a CSV of customers and download predictions |
| **Model Info** | View confusion matrix, ROC curve, threshold analysis, and model summary |

---

## 🧠 Explainability with SHAP

The app uses **SHAP (SHapley Additive exPlanations)** to explain individual predictions:

- **Red bars** → features pushing toward churn
- **Green bars** → features pushing against churn
- The top 10 most influential features are shown for every prediction

---

## 🗃️ Dataset

**IBM Telco Customer Churn** — 7,043 customers, 21 features including:
- Demographics: gender, SeniorCitizen, Partner, Dependents
- Account: tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
- Services: PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies

**Class imbalance:** ~26.5% churn rate — handled via threshold tuning (0.3).

---

## 🛠️ Technologies

- **Python 3.10+**
- **Pandas, NumPy** — data manipulation
- **Matplotlib, Seaborn** — visualisation
- **Scikit-learn** — preprocessing & modelling
- **SHAP** — explainable AI
- **Streamlit** — interactive dashboard
- **Git & GitHub** — version control

---

## 📬 Contact

**Greeshma** · [greeshmaa965@gmail.com](mailto:greeshmaa965@gmail.com)

GitHub: [@greeshmaa965-cell](https://github.com/greeshmaa965-cell)