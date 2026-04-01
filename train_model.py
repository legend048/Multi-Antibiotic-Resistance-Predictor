"""
AMR Multi-Task Predictor — Training Script
Trains a MultiOutputClassifier (RandomForest) on curated E. coli AMR genomic data.
Data was sourced from PATRIC/BV-BRC, cleaned, and pre-processed before training.
Saves model, encoders, and evaluation metrics as JSON artifacts.
"""

import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, matthews_corrcoef, classification_report,
    roc_auc_score, f1_score
)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH   = "data/amr_dataset.csv"
MODEL_DIR   = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

ANTIBIOTICS = ["CIPRO", "CEFTRIAXONE", "AMOXICILLIN"]
FEATURE_COLS = [f"F{i}" for i in range(1, 41)]

print("=" * 60)
print("  AMR Multi-Task Resistance Predictor — Training")
print("=" * 60)

# ── Load Data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"\n✅ Loaded dataset: {df.shape[0]} genomes × {df.shape[1]} columns")

X = df[FEATURE_COLS].values

# ── Encode Labels ─────────────────────────────────────────────────────────────
encoders = {}
Y = np.zeros((len(df), len(ANTIBIOTICS)), dtype=int)
for i, ab in enumerate(ANTIBIOTICS):
    le = LabelEncoder()
    le.fit(["S", "I", "R"])          # fixed order
    Y[:, i] = le.transform(df[ab])
    encoders[ab] = le
    dist = df[ab].value_counts(normalize=True).to_dict()
    print(f"   {ab:15s}: S={dist.get('S',0):.1%}  I={dist.get('I',0):.1%}  R={dist.get('R',0):.1%}")

# ── Train / Test Split ────────────────────────────────────────────────────────
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=42, stratify=Y[:, 0]
)
print(f"\n📊 Train: {len(X_train)} samples   Test: {len(X_test)} samples")

# ── Train Model ───────────────────────────────────────────────────────────────
base = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=4,
    n_jobs=-1,
    random_state=42
)
model = MultiOutputClassifier(base, n_jobs=-1)
model.fit(X_train, Y_train)
print("\n✅ Model trained successfully!")

# ── Evaluate ──────────────────────────────────────────────────────────────────
Y_pred = model.predict(X_test)
metrics = {}

print("\n" + "─" * 60)
print("  Per-Antibiotic Performance")
print("─" * 60)

for i, ab in enumerate(ANTIBIOTICS):
    y_true = Y_test[:, i]
    y_pred = Y_pred[:, i]

    acc  = accuracy_score(y_true, y_pred)
    mcc  = matthews_corrcoef(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="weighted")

    # AUC (one-vs-rest)
    proba = model.estimators_[i].predict_proba(X_test)
    try:
        auc = roc_auc_score(y_true, proba, multi_class="ovr", average="weighted")
    except Exception:
        auc = 0.0

    metrics[ab] = {"accuracy": round(acc,4), "mcc": round(mcc,4),
                   "f1": round(f1,4), "auc": round(auc,4)}
    print(f"  {ab:15s}  Acc={acc:.3f}  MCC={mcc:.3f}  F1={f1:.3f}  AUC={auc:.3f}")

    # Confusion-style class accuracy
    report = classification_report(y_true, y_pred,
                                   target_names=encoders[ab].classes_,
                                   output_dict=True)
    for cls in ["S","I","R"]:
        if cls in report:
            metrics[ab][f"precision_{cls}"] = round(report[cls]["precision"], 3)
            metrics[ab][f"recall_{cls}"]    = round(report[cls]["recall"], 3)

# ── Feature Importances (avg across estimators) ───────────────────────────────
importances = np.mean(
    [est.feature_importances_ for est in model.estimators_], axis=0
)
fi_dict = {f"F{i+1}": round(float(importances[i]), 5) for i in range(40)}
top10 = sorted(fi_dict.items(), key=lambda x: -x[1])[:10]
print("\n  Top 10 Features (avg importance):")
for feat, imp in top10:
    bar = "█" * int(imp * 400)
    print(f"    {feat:5s} {imp:.4f}  {bar}")

per_ab_fi = {}
for i, ab in enumerate(ANTIBIOTICS):
    fi = model.estimators_[i].feature_importances_
    top = sorted(
        {f"F{j+1}": round(float(fi[j]),5) for j in range(40)}.items(),
        key=lambda x: -x[1]
    )[:5]
    per_ab_fi[ab] = [{"feature": k, "importance": v} for k,v in top]

# ── Save Artifacts ────────────────────────────────────────────────────────────
joblib.dump(model,    f"{MODEL_DIR}/amr_model.pkl")
joblib.dump(encoders, f"{MODEL_DIR}/encoders.pkl")

with open(f"{MODEL_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

with open(f"{MODEL_DIR}/feature_importance.json", "w") as f:
    json.dump({"overall": fi_dict, "per_antibiotic": per_ab_fi}, f, indent=2)

with open(f"{MODEL_DIR}/feature_cols.json", "w") as f:
    json.dump(FEATURE_COLS, f)

print("\n✅ All artifacts saved to /models/")
print("   amr_model.pkl | encoders.pkl | metrics.json | feature_importance.json")
print("=" * 60)
