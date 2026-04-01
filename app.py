"""
AMR Multi-Task Resistance Predictor — Flask Web Application
"""

from flask import Flask, render_template, request, jsonify
import joblib
import json
import numpy as np
import os
import io
import re
import urllib.error
import urllib.request

app = Flask(__name__)


def _load_env_file(env_path=".env"):
    """Load key/value pairs from a local .env file without overriding existing env vars."""
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as env_file:
        for line in env_file:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue

            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"").strip("'")
            if key:
                os.environ.setdefault(key, value)


_load_env_file()

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


def _extract_text_from_uploaded_file(uploaded_file):
    """Extract UTF-8 text from text-like files and basic text extraction from PDFs."""
    file_name = (uploaded_file.filename or "").lower()
    raw_bytes = uploaded_file.read()

    if not raw_bytes:
        raise ValueError("Uploaded file is empty.")

    if len(raw_bytes) > 2 * 1024 * 1024:
        raise ValueError("Uploaded file is too large. Maximum supported size is 2 MB.")

    if file_name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ValueError("PDF upload requires pypdf. Install dependencies or upload TXT/CSV/JSON.")

        reader = PdfReader(io.BytesIO(raw_bytes))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    else:
        text = raw_bytes.decode("utf-8", errors="ignore")

    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("No readable text was found in the uploaded file.")

    # Keep prompt size bounded to avoid token overflow and slow responses.
    return cleaned_text[:12000]


def _extract_json_from_llm_response(raw_text):
    """Extract and parse the first JSON object from LLM output."""
    stripped = raw_text.strip()
    if stripped.startswith("```"):
        stripped = stripped.replace("```json", "").replace("```", "").strip()

    match = re.search(r"\{[\s\S]*\}", stripped)
    if not match:
        raise ValueError("LLM output did not contain a valid JSON object.")

    return json.loads(match.group(0))


def _normalize_feature_payload(raw_features):
    """Ensure all expected features exist and are numeric within [0, 1]."""
    normalized = {}
    for feature in FEATURE_COLS:
        try:
            value = float(raw_features.get(feature, 0))
        except (TypeError, ValueError):
            value = 0.0

        value = max(0.0, min(1.0, value))
        normalized[feature] = round(value, 4)

    return normalized


def _strip_model_prefix(model_name):
    if model_name.startswith("models/"):
        return model_name.split("models/", 1)[1]
    return model_name


def _list_generate_content_models(api_key):
    """Return model names that support generateContent for this API key."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    req = urllib.request.Request(url, method="GET")

    with urllib.request.urlopen(req, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))

    models = []
    for item in data.get("models", []):
        name = item.get("name", "")
        methods = item.get("supportedGenerationMethods", [])
        if name and "generateContent" in methods:
            models.append(name)

    return models


def _resolve_generate_model_id(api_key, requested_model):
    """Pick a valid model for generateContent, with graceful fallback."""
    available_models = _list_generate_content_models(api_key)
    if not available_models:
        raise ValueError("No AI Studio models available for generateContent with this API key.")

    requested_short = _strip_model_prefix((requested_model or "").strip())
    available_short = [_strip_model_prefix(name) for name in available_models]

    if requested_short in available_short:
        return requested_short

    # Accept close aliases such as gemini-2.0-flash vs gemini-2.0-flash-001.
    if requested_short:
        for model_short in available_short:
            if model_short.startswith(f"{requested_short}-") or requested_short.startswith(f"{model_short}-"):
                return model_short

    preferred_order = [
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-flash-latest",
        "gemini-2.0-flash-lite",
        "gemini-2.5-flash-lite",
        "gemini-pro-latest"
    ]
    for preferred in preferred_order:
        if preferred in available_short:
            return preferred

    for model_short in available_short:
        if (
            model_short.startswith("gemini")
            and "tts" not in model_short
            and "image" not in model_short
            and "computer-use" not in model_short
            and "robotics" not in model_short
        ):
            return model_short

    return available_short[0]

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


@app.route("/extract_features", methods=["POST"])
def extract_features():
    api_key = request.form.get("api_key", "").strip()
    if not api_key:
        api_key = (
            os.getenv("ai_studio_key", "").strip()
            or os.getenv("AI_STUDIO_KEY", "").strip()
            or os.getenv("GOOGLE_API_KEY", "").strip()
        )

    selected_model = request.form.get("model", "gemini-2.0-flash").strip()
    uploaded_file = request.files.get("medical_file")

    if not api_key:
        return jsonify({"error": "AI Studio API key is missing. Add it in .env as ai_studio_key or pass it in the form."}), 400

    if not uploaded_file or not uploaded_file.filename:
        return jsonify({"error": "Please upload a medical data file."}), 400

    try:
        model_name = _resolve_generate_model_id(api_key, selected_model)
        medical_text = _extract_text_from_uploaded_file(uploaded_file)
        prompt = f"""
You are extracting AMR genomic features for a resistance model.

Task:
1) Read the medical/genomics text below.
2) Infer numeric values for each feature key.
3) Return ONLY a JSON object with exactly these keys: {", ".join(FEATURE_COLS)}

Rules:
- Values must be numeric and in range [0, 1].
- If there is not enough evidence, use 0.
- Do not include explanations, markdown, or extra keys.

Medical data:
{medical_text}
"""

        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json"
            }
        }

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=45) as response:
            api_data = json.loads(response.read().decode("utf-8"))

        candidates = api_data.get("candidates", [])
        if not candidates:
            raise ValueError("AI Studio returned no candidate response.")

        parts = candidates[0].get("content", {}).get("parts", [])
        llm_text = "\n".join(part.get("text", "") for part in parts if "text" in part).strip()
        if not llm_text:
            raise ValueError("AI Studio returned an empty response.")

        extracted = _extract_json_from_llm_response(llm_text)
        if not isinstance(extracted, dict):
            raise ValueError("AI Studio response format is invalid. Expected a JSON object.")

        normalized_features = _normalize_feature_payload(extracted)
        non_zero_count = sum(1 for val in normalized_features.values() if val > 0)

        return jsonify({
            "features": normalized_features,
            "model": f"models/{model_name}",
            "non_zero_features": non_zero_count
        })

    except urllib.error.HTTPError as err:
        error_body = err.read().decode("utf-8", errors="ignore")
        try:
            parsed = json.loads(error_body)
            message = parsed.get("error", {}).get("message", "AI Studio request failed.")
        except json.JSONDecodeError:
            message = error_body[:250] or "AI Studio request failed."
        return jsonify({"error": message}), err.code if 400 <= err.code < 600 else 502

    except urllib.error.URLError:
        return jsonify({"error": "Unable to reach AI Studio API. Check network and try again."}), 502

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    except Exception as exc:
        return jsonify({"error": f"Unable to extract genomic features: {exc}"}), 500

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
