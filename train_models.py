"""
train_models.py -- Train all ensemble models and save them for the Streamlit prototype.

This script:
1. Downloads the credit card fraud dataset from Kaggle
2. Preprocesses (scales Amount and Time)
3. Applies SMOTE for class balancing
4. Trains RandomForest, XGBoost, VotingClassifier, and StackingClassifier
5. Saves models, scalers, test data, and evaluation metrics to disk
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


def main():
    # ----------------------------------------------
    # 1. Load Data
    # ----------------------------------------------
    print("[*] Downloading dataset...")
    import kagglehub
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    csv_file = [f for f in os.listdir(path) if f.endswith(".csv")][0]
    df = pd.read_csv(os.path.join(path, csv_file))
    print(f"   Dataset shape: {df.shape}")

    # ----------------------------------------------
    # 2. Preprocess
    # ----------------------------------------------
    print("[*] Preprocessing...")
    amount_scaler = StandardScaler()
    time_scaler = StandardScaler()

    df["Amount_scaled"] = amount_scaler.fit_transform(df[["Amount"]])
    df["Time_scaled"] = time_scaler.fit_transform(df[["Time"]])
    df = df.drop(["Amount", "Time"], axis=1)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"   Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    print(f"   Fraud in train: {(y_train == 1).sum()} | Fraud in test: {(y_test == 1).sum()}")

    # ----------------------------------------------
    # 3. SMOTE
    # ----------------------------------------------
    print("[*] Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"   After SMOTE -- Class 0: {(y_train_smote == 0).sum()} | Class 1: {(y_train_smote == 1).sum()}")

    # ----------------------------------------------
    # 4. Train Models
    # ----------------------------------------------
    models = {}
    metrics = {}

    # --- Random Forest ---
    print("\n[*] Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    start = time.time()
    rf.fit(X_train_smote, y_train_smote)
    rf_time = time.time() - start
    print(f"   Training time: {rf_time:.2f}s")
    models["Random Forest"] = rf

    # --- XGBoost ---
    print("[*] Training XGBoost...")
    xgb = XGBClassifier(random_state=42, eval_metric="logloss")
    start = time.time()
    xgb.fit(X_train_smote, y_train_smote)
    xgb_time = time.time() - start
    print(f"   Training time: {xgb_time:.2f}s")
    models["XGBoost"] = xgb

    # --- Logistic Regression (for ensembles) ---
    print("[*] Training Logistic Regression...")
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    start = time.time()
    log_reg.fit(X_train_smote, y_train_smote)
    lr_time = time.time() - start
    print(f"   Training time: {lr_time:.2f}s")
    models["Logistic Regression"] = log_reg

    # --- Voting Classifier ---
    print("[*] Training Voting Classifier (RF + XGB + LR)...")
    voting_clf = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb), ("lr", log_reg)],
        voting="soft",
    )
    start = time.time()
    voting_clf.fit(X_train_smote, y_train_smote)
    voting_time = time.time() - start
    print(f"   Training time: {voting_time:.2f}s")
    models["Voting Classifier"] = voting_clf

    # --- Stacking Classifier ---
    print("[*] Training Stacking Classifier (RF + XGB -> LR)...")
    stacking_clf = StackingClassifier(
        estimators=[("rf", rf), ("xgb", xgb)],
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5,
    )
    start = time.time()
    stacking_clf.fit(X_train_smote, y_train_smote)
    stacking_time = time.time() - start
    print(f"   Training time: {stacking_time:.2f}s")
    models["Stacking Classifier"] = stacking_clf

    # ----------------------------------------------
    # 5. Evaluate All Models
    # ----------------------------------------------
    print("\n[*] Evaluating models...")
    training_times = {
        "Random Forest": rf_time,
        "XGBoost": xgb_time,
        "Logistic Regression": lr_time,
        "Voting Classifier": voting_time,
        "Stacking Classifier": stacking_time,
    }

    for name, model in models.items():
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        cm = confusion_matrix(y_test, preds).tolist()

        precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, probs)

        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds)),
            "recall": float(recall_score(y_test, preds)),
            "f1": float(f1_score(y_test, preds)),
            "auprc": float(average_precision_score(y_test, probs)),
            "roc_auc": float(roc_auc_score(y_test, probs)),
            "confusion_matrix": cm,
            "training_time": training_times[name],
        }
        print(f"\n=== {name} ===")
        print(classification_report(y_test, preds, target_names=["Legitimate", "Fraud"]))

    # ----------------------------------------------
    # 6. Save Everything
    # ----------------------------------------------
    output_dir = os.path.join(os.path.dirname(__file__), "saved_artifacts")
    os.makedirs(output_dir, exist_ok=True)

    # Save models
    for name, model in models.items():
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(model, os.path.join(output_dir, f"{safe_name}.pkl"))
    print(f"\n[+] Models saved to {output_dir}/")

    # Save scalers
    joblib.dump(amount_scaler, os.path.join(output_dir, "amount_scaler.pkl"))
    joblib.dump(time_scaler, os.path.join(output_dir, "time_scaler.pkl"))
    print("[+] Scalers saved")

    # Save metrics
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("[+] Metrics saved")

    # Save feature names
    with open(os.path.join(output_dir, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)
    print("[+] Feature names saved")

    # Save demo data (mix of fraud + legit from test set)
    fraud_sample = X_test[y_test == 1].sample(min(50, (y_test == 1).sum()), random_state=42)
    legit_sample = X_test[y_test == 0].sample(150, random_state=42)
    demo_data = pd.concat([fraud_sample, legit_sample])
    demo_labels = pd.concat([y_test.loc[fraud_sample.index], y_test.loc[legit_sample.index]])
    demo_df = demo_data.copy()
    demo_df["Actual_Class"] = demo_labels.values
    demo_df.to_csv(os.path.join(output_dir, "demo_transactions.csv"), index=False)
    print("[+] Demo transactions saved")

    # Save dataset stats
    dataset_stats = {
        "total_transactions": int(df.shape[0]),
        "total_features": int(X.shape[1]),
        "total_legitimate": int((y == 0).sum()),
        "total_fraud": int((y == 1).sum()),
        "fraud_percentage": float((y == 1).sum() / len(y) * 100),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "train_fraud": int((y_train == 1).sum()),
        "test_fraud": int((y_test == 1).sum()),
        "smote_class_0": int((y_train_smote == 0).sum()),
        "smote_class_1": int((y_train_smote == 1).sum()),
    }
    with open(os.path.join(output_dir, "dataset_stats.json"), "w") as f:
        json.dump(dataset_stats, f, indent=2)
    print("[+] Dataset stats saved")

    print("\n[OK] All done! Run `streamlit run app.py` to launch the prototype.")


if __name__ == "__main__":
    main()
