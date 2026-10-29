# 🛡️ Credit Card Fraud Detection — Ensemble ML Prototype

An interactive web-based prototype for detecting fraudulent credit card transactions using **ensemble machine learning techniques**.

## 📊 Dataset

- **Source**: [ULB Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions with 31 features
- **Features**: V1–V28 (PCA-transformed), Time, Amount, Class
- **Class Distribution**: 99.83% Legitimate, 0.17% Fraud (highly imbalanced)

## 🤖 Models

| Model | Description |
|-------|-------------|
| **Random Forest** | 100 trees, bagging-based ensemble |
| **XGBoost** | Gradient boosting with regularization |
| **Logistic Regression** | Linear baseline for ensembles |
| **Voting Classifier** | Soft voting: RF + XGB + LR |
| **Stacking Classifier** | RF + XGB as base learners → LR as meta-learner |

## 🔧 Pipeline

1. **Data Loading** — Download from Kaggle via `kagglehub`
2. **Preprocessing** — StandardScaler on `Amount` and `Time`
3. **SMOTE** — Synthetic Minority Oversampling for class balance
4. **Training** — Train 5 models on balanced data
5. **Evaluation** — Precision, Recall, F1, AUPRC, ROC-AUC

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train models
```bash
python train_models.py
```
This downloads the dataset, trains all 5 models, and saves:
- Trained model files (`.pkl`)
- Scalers, metrics, feature names
- Demo transaction data for the app

### 3. Launch the prototype
```bash
streamlit run app.py
```

## 📱 Prototype Features

### 🏠 Overview
- Dataset statistics and class distribution
- Pipeline architecture visualization
- Model performance summary table

### 📊 Model Comparison
- Side-by-side metric bar charts
- Confusion matrices for all models
- Radar chart for multi-dimensional comparison
- Training time comparison

### 🔍 Live Prediction
- Select any trained model
- Load random demo transactions or enter manual input
- Real-time fraud probability gauge
- Side-by-side prediction vs. actual comparison

### 📋 Batch Analysis
- Process multiple transactions at once
- Upload custom CSV or use demo data
- Batch evaluation metrics
- Downloadable results CSV

## 📁 Project Structure

```
credit-card-fraud-detection/
├── app.py                 # Streamlit prototype
├── train_models.py        # Model training script
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── saved_artifacts/       # Generated after training
    ├── random_forest.pkl
    ├── xgboost.pkl
    ├── logistic_regression.pkl
    ├── voting_classifier.pkl
    ├── stacking_classifier.pkl
    ├── amount_scaler.pkl
    ├── time_scaler.pkl
    ├── metrics.json
    ├── feature_names.json
    ├── dataset_stats.json
    └── demo_transactions.csv
```
