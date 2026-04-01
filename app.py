"""
AMR Multi-Task Resistance Predictor — Flask Web Application
"""

from flask import Flask, render_template, request, jsonify
import joblib
import json
import numpy as np
import os

app = Flask(__name__)

# ── Load Model Artifacts ──────────────────────────────────────────────────────
MODEL_DIR   = "models"
model       = joblib.load(f"{MODEL_DIR}/amr_model.pkl")
encoders    = joblib.load(f"{MODEL_DIR}/encoders.pkl")
FEATURE_COLS = json.load(open(f"{MODEL_DIR}/feature_cols.json"))
METRICS     = json.load(open(f"{MODEL_DIR}/metrics.json"))
FI          = json.load(open(f"{MODEL_DIR}/feature_importance.json"))
ANTIBIOTICS = ["CIPRO", "CEFTRIAXONE", "AMOXICILLIN"]

LABEL_MAP   = {0: "S", 1: "I", 2: "R"}
LABEL_FULL  = {"S": "Susceptible", "I": "Intermediate", "R": "Resistant"}
LABEL_COLOR = {"S": "#00d4aa", "I": "#f5a623", "R": "#ff4d6d"}

ANTIBIOTIC_INFO = {
    "CIPRO": {
        "full": "Ciprofloxacin",
        "family": "Fluoroquinolone",
        "treats": "UTIs, respiratory & GI infections",
        "mechanism": "Blocks bacterial DNA replication"
    },
    "CEFTRIAXONE": {
        "full": "Ceftriaxone",
        "family": "3rd-gen Cephalosporin",
        "treats": "Pneumonia, meningitis, severe hospital infections",
        "mechanism": "Destroys bacterial cell wall"
    },
    "AMOXICILLIN": {
        "full": "Amoxicillin",
        "family": "Penicillin",
        "treats": "Ear infections, strep throat, chest infections",
        "mechanism": "Disrupts cell wall synthesis"
    }
}

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",
                           antibiotics=ANTIBIOTICS,
                           antibiotic_info=ANTIBIOTIC_INFO,
                           metrics=METRICS,
                           feature_importance=FI["per_antibiotic"],
                           feature_cols=FEATURE_COLS)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data.get("features", {})

    # Build feature vector (fill missing with 0)
    X = np.array([[float(features.get(f, 0)) for f in FEATURE_COLS]])

    predictions = {}
    for i, ab in enumerate(ANTIBIOTICS):
        estimator = model.estimators_[i]
        pred_idx  = estimator.predict(X)[0]
        proba     = estimator.predict_proba(X)[0]
        label     = LABEL_MAP[pred_idx]

        # Probabilities for S, I, R (class order from encoder)
        classes   = list(estimator.classes_)
        prob_dict = {}
        for cls_idx, cls_label in LABEL_MAP.items():
            if cls_idx in classes:
                prob_dict[cls_label] = round(float(proba[classes.index(cls_idx)]), 3)
            else:
                prob_dict[cls_label] = 0.0

        predictions[ab] = {
            "label":      label,
            "full":       LABEL_FULL[label],
            "color":      LABEL_COLOR[label],
            "confidence": round(max(proba) * 100, 1),
            "probabilities": prob_dict,
            "info":       ANTIBIOTIC_INFO[ab]
        }

    # Top contributing features
    overall_fi = FI["overall"]
    top_features = sorted(overall_fi.items(), key=lambda x: -x[1])[:8]
    contributing = [
        {"feature": k, "value": float(features.get(k, 0)),
         "importance": v, "active": float(features.get(k, 0)) > 0}
        for k, v in top_features
    ]

    return jsonify({"predictions": predictions, "contributing": contributing})

@app.route("/metrics")
def get_metrics():
    return jsonify({"metrics": METRICS, "feature_importance": FI})

@app.route("/random_sample")
def random_sample():
    """Return a random realistic genome sample."""
    import pandas as pd
    df = pd.read_csv("data/amr_dataset.csv")
    row = df.sample(1).iloc[0]
    features = {f: float(row[f]) for f in FEATURE_COLS}
    actual = {ab: row[ab] for ab in ANTIBIOTICS}
    return jsonify({"features": features, "actual": actual})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
