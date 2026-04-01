"""
AMR Multi-Task Resistance Predictor — Flask Web Application
"""

from flask import Flask, render_template, request, jsonify
import joblib
import json
import numpy as np
import pandas as pd
import os
import io
import re
import socket
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
CORE_ANTIBIOTICS = ["CIPRO", "CEFTRIAXONE", "AMOXICILLIN"]

ONLINE_PHENO_URL = (
    "https://raw.githubusercontent.com/Arcadia-Science/"
    "2024-Ecoli-amr-genotype-phenotype_7000strains/main/"
    "dataset_analysis/results/phenotype_matrix_08302024.csv"
)
ONLINE_PHENO_PATH = "data/external/phenotype_matrix_08302024.csv"

CORE_TO_ONLINE_COLUMN = {
    "CIPRO": "ciprofloxacin",
    "CEFTRIAXONE": "ceftriaxone",
    "AMOXICILLIN": "amoxicillin"
}

EXTRA_TO_ONLINE_COLUMN = {
    "GENTAMICIN": "gentamicin",
    "CEFTAZIDIME": "ceftazidime",
    "PIPERACILLIN_TAZOBACTAM": "piperacillin/tazobactam",
    "MEROPENEM": "meropenem",
    "TRIMETHOPRIM_SULFAMETHOXAZOLE": "trimethoprim/sulfamethoxazole"
}

LABEL_MAP   = {0: "S", 1: "I", 2: "R"}
LABEL_FULL  = {"S": "Susceptible", "I": "Intermediate", "R": "Resistant"}
LABEL_COLOR = {"S": "#00d4aa", "I": "#f5a623", "R": "#ff4d6d"}
AST_LABELS  = ("S", "I", "R")

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
    },
    "GENTAMICIN": {
        "full": "Gentamicin",
        "family": "Aminoglycoside",
        "treats": "Severe Gram-negative infections and sepsis",
        "mechanism": "Inhibits bacterial protein synthesis"
    },
    "CEFTAZIDIME": {
        "full": "Ceftazidime",
        "family": "3rd-gen Cephalosporin",
        "treats": "Serious hospital-acquired Gram-negative infections",
        "mechanism": "Inhibits bacterial cell wall synthesis"
    },
    "PIPERACILLIN_TAZOBACTAM": {
        "full": "Piperacillin/Tazobactam",
        "family": "Ureidopenicillin + beta-lactamase inhibitor",
        "treats": "Complicated intra-abdominal and polymicrobial infections",
        "mechanism": "Cell wall inhibition with beta-lactamase protection"
    },
    "MEROPENEM": {
        "full": "Meropenem",
        "family": "Carbapenem",
        "treats": "Severe multidrug-resistant bacterial infections",
        "mechanism": "Broad-spectrum cell wall synthesis inhibition"
    },
    "TRIMETHOPRIM_SULFAMETHOXAZOLE": {
        "full": "Trimethoprim/Sulfamethoxazole",
        "family": "Folate pathway inhibitor combination",
        "treats": "UTIs and opportunistic bacterial infections",
        "mechanism": "Blocks bacterial folate synthesis at two steps"
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


def _normalize_ast_label(raw_value):
    if raw_value is None or pd.isna(raw_value):
        return None

    value = str(raw_value).strip().lower()
    if value in {"s", "susceptible"}:
        return "S"
    if value in {"i", "intermediate"}:
        return "I"
    if value in {"r", "resistant", "non-susceptible", "nonsusceptible"}:
        return "R"

    if "intermediate" in value:
        return "I"
    if "susceptible" in value and "non-susceptible" not in value and "nonsusceptible" not in value:
        return "S"
    if "resistant" in value or "non-susceptible" in value or "nonsusceptible" in value:
        return "R"
    return None


def _ensure_online_phenotype_file():
    if os.path.exists(ONLINE_PHENO_PATH):
        return ONLINE_PHENO_PATH

    os.makedirs(os.path.dirname(ONLINE_PHENO_PATH), exist_ok=True)
    try:
        req = urllib.request.Request(ONLINE_PHENO_URL, method="GET")
        with urllib.request.urlopen(req, timeout=45) as response:
            data = response.read()
        if not data:
            return None

        with open(ONLINE_PHENO_PATH, "wb") as out_file:
            out_file.write(data)
        return ONLINE_PHENO_PATH
    except Exception:
        return None


def _safe_distribution(values):
    total = sum(values.values())
    if total <= 0:
        return {label: 0.0 for label in AST_LABELS}
    return {label: round(values.get(label, 0) / total, 6) for label in AST_LABELS}


def _build_online_extension_model():
    source_path = _ensure_online_phenotype_file()
    if not source_path or not os.path.exists(source_path):
        return {}

    try:
        df = pd.read_csv(source_path)
    except Exception:
        return {}

    column_lookup = {str(col).strip().lower(): col for col in df.columns}

    core_columns = {}
    for core_ab, online_col in CORE_TO_ONLINE_COLUMN.items():
        actual_col = column_lookup.get(online_col.lower())
        if actual_col:
            core_columns[core_ab] = actual_col

    extension_model = {}
    for target_ab, target_online_col in EXTRA_TO_ONLINE_COLUMN.items():
        target_col = column_lookup.get(target_online_col.lower())
        if not target_col:
            continue

        target_series = df[target_col].map(_normalize_ast_label)
        overall_counts = target_series.value_counts().to_dict()
        overall_dist = _safe_distribution(overall_counts)

        per_source = {}
        total_support = 0
        for source_ab, source_col in core_columns.items():
            subset = df[[source_col, target_col]].dropna()
            if subset.empty:
                continue

            subset = subset.copy()
            subset[source_col] = subset[source_col].map(_normalize_ast_label)
            subset[target_col] = subset[target_col].map(_normalize_ast_label)
            subset = subset.dropna()
            if subset.empty:
                continue

            source_distributions = {}
            for source_label in AST_LABELS:
                source_slice = subset[subset[source_col] == source_label]
                support = int(len(source_slice))
                if support < 30:
                    continue

                target_counts = source_slice[target_col].value_counts().to_dict()
                dist = _safe_distribution(target_counts)
                dist["support"] = support
                source_distributions[source_label] = dist
                total_support += support

            if source_distributions:
                per_source[source_ab] = source_distributions

        if per_source or any(overall_dist.values()):
            extension_model[target_ab] = {
                "sources": per_source,
                "overall": overall_dist,
                "total_support": total_support
            }

    return extension_model


def _infer_extended_predictions(core_predictions):
    inferred = {}
    if not ONLINE_EXTENSION_MODEL:
        return inferred

    for target_ab, target_model in ONLINE_EXTENSION_MODEL.items():
        combined = {label: 0.0 for label in AST_LABELS}
        total_weight = 0.0
        used_support = 0

        for source_ab, source_stats in target_model.get("sources", {}).items():
            source_pred = core_predictions.get(source_ab)
            if not source_pred:
                continue

            source_label = source_pred["label"]
            label_dist = source_stats.get(source_label)
            if not label_dist:
                continue

            source_weight = source_pred["probabilities"].get(source_label, 0.0)
            source_weight = max(source_weight, 0.05)

            for cls_label in AST_LABELS:
                combined[cls_label] += source_weight * float(label_dist.get(cls_label, 0.0))

            total_weight += source_weight
            used_support += int(label_dist.get("support", 0))

        if total_weight > 0:
            for cls_label in AST_LABELS:
                combined[cls_label] = combined[cls_label] / total_weight
        else:
            combined = dict(target_model.get("overall", {label: 0.0 for label in AST_LABELS}))
            used_support = int(target_model.get("total_support", 0))

        pred_label = max(combined, key=combined.get)
        base_conf = float(combined[pred_label]) * 100
        support_factor = min(1.0, used_support / 800.0) if used_support > 0 else 0.5
        confidence = round(base_conf * max(support_factor, 0.5), 1)

        inferred[target_ab] = {
            "label": pred_label,
            "full": LABEL_FULL[pred_label],
            "color": LABEL_COLOR[pred_label],
            "confidence": confidence,
            "probabilities": {label: round(float(combined.get(label, 0.0)), 3) for label in AST_LABELS},
            "info": ANTIBIOTIC_INFO[target_ab],
            "derived": True,
            "support": used_support
        }

    return inferred


ONLINE_EXTENSION_MODEL = _build_online_extension_model()
SUPPORTED_ANTIBIOTICS = CORE_ANTIBIOTICS + [ab for ab in ONLINE_EXTENSION_MODEL.keys() if ab not in CORE_ANTIBIOTICS]


def _extract_explicit_feature_values(medical_text):
    """Parse explicit F1-F40 numeric assignments if provided in the report."""
    feature_pattern = re.compile(
        r"\b(F(?:[1-9]|[12][0-9]|3[0-9]|40))\s*[:=]\s*(-?\d+(?:\.\d+)?)\b",
        re.IGNORECASE
    )
    extracted = {}
    for feature, raw_value in feature_pattern.findall(medical_text):
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue

        extracted[feature.upper()] = round(max(0.0, min(1.0, value)), 4)

    return extracted


def _detect_entity_signal(lower_text, entity_patterns):
    """Detect positive/negative mention around genomic marker phrases."""
    positive_terms = ["detected", "present", "positive", "identified", "found", "elevated", "high"]
    negative_terms = ["not detected", "absent", "negative", "undetected", "not found", "no evidence"]

    for pattern in entity_patterns:
        for match in re.finditer(pattern, lower_text):
            start = max(0, match.start() - 90)
            end = min(len(lower_text), match.end() + 90)
            window = lower_text[start:end]

            if any(term in window for term in negative_terms):
                return 0.0
            if any(term in window for term in positive_terms):
                return 1.0

            # Mentioned marker but uncertain context.
            return 0.7

    return None


def _extract_seed_features_from_medical_text(medical_text):
    """Build seed features from direct report clues so narrative reports are usable."""
    explicit_features = _extract_explicit_feature_values(medical_text)
    seed_features = dict(explicit_features)
    lower_text = medical_text.lower()

    marker_map = {
        "F1": [r"gyra\s*s83l", r"\bs83l\b"],
        "F2": [r"gyra\s*d87g", r"\bd87g\b"],
        "F3": [r"bla[_-]?tem", r"\btem\b"],
        "F7": [r"bla[_-]?shv", r"\bshv\b"],
        "F11": [r"bla[_-]?ctx[- ]?m[- ]?15", r"ctx[- ]?m[- ]?15", r"\besbl\b"]
    }

    for feature, patterns in marker_map.items():
        if feature in seed_features:
            continue

        signal = _detect_entity_signal(lower_text, patterns)
        if signal is not None:
            seed_features[feature] = signal

    if "F4" not in seed_features:
        if re.search(r"mdr\s+plasmid|multi[- ]drug resistance plasmid|multidrug plasmid|plasmid marker", lower_text):
            seed_features["F4"] = 0.9
        elif "plasmid" in lower_text and ("resistance" in lower_text or "mdr" in lower_text):
            seed_features["F4"] = 0.65

    if "F16" not in seed_features and "efflux pump" in lower_text:
        if any(term in lower_text for term in ["elevated", "high", "overexpressed", "upregulated"]):
            seed_features["F16"] = 0.8
        elif any(term in lower_text for term in ["moderate", "increased"]):
            seed_features["F16"] = 0.6
        elif any(term in lower_text for term in ["low", "minimal"]):
            seed_features["F16"] = 0.25
        else:
            seed_features["F16"] = 0.5

    high_risk_phrases = [
        "multi-drug resistance",
        "multidrug resistance",
        "likely resistant",
        "resistant phenotype",
        "high amr burden",
        "extensively drug resistant",
        "esbl"
    ]
    low_risk_phrases = [
        "low risk",
        "susceptible",
        "no major resistance",
        "no resistance markers",
        "limited amr signal"
    ]

    risk_score = 0
    for phrase in high_risk_phrases:
        if phrase in lower_text:
            risk_score += 1
    for phrase in low_risk_phrases:
        if phrase in lower_text:
            risk_score -= 1

    if any(feat in seed_features and seed_features[feat] > 0.5 for feat in ["F1", "F2", "F3", "F7", "F11"]):
        risk_score += 1

    baseline_prop = max(0.0, min(1.0, 0.35 + 0.08 * risk_score))
    for idx in range(31, 41):
        feat = f"F{idx}"
        if feat in seed_features:
            continue

        offset = ((idx - 31) % 4 - 1.5) * 0.02
        seed_features[feat] = round(max(0.0, min(1.0, baseline_prop + offset)), 4)

    return seed_features, explicit_features


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


def _build_generation_payload(prompt_text, enforce_json_mode=True):
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
        "generationConfig": {
            "temperature": 0
        }
    }
    if enforce_json_mode:
        payload["generationConfig"]["responseMimeType"] = "application/json"
    return payload


def _call_generate_content(api_key, model_name, prompt_text):
    """Call generateContent and gracefully fallback if strict JSON mode is unsupported."""
    request_timeout = 90
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    payload = _build_generation_payload(prompt_text, enforce_json_mode=True)
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=request_timeout) as response:
            return json.loads(response.read().decode("utf-8")), True
    except urllib.error.HTTPError as err:
        error_body = err.read().decode("utf-8", errors="ignore")
        lowered = error_body.lower()
        json_mode_unsupported = (
            err.code == 400
            and (
                "json mode is not enabled" in lowered
                or (
                    "responsemimetype" in lowered
                    and ("not enabled" in lowered or "unsupported" in lowered)
                )
            )
        )

        if not json_mode_unsupported:
            raise urllib.error.HTTPError(err.url, err.code, err.reason, err.headers, io.BytesIO(error_body.encode("utf-8")))

        fallback_payload = _build_generation_payload(prompt_text, enforce_json_mode=False)
        fallback_req = urllib.request.Request(
            url,
            data=json.dumps(fallback_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(fallback_req, timeout=request_timeout) as response:
            return json.loads(response.read().decode("utf-8")), False

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    avg_accuracy = round(sum(METRICS[ab]["accuracy"] for ab in CORE_ANTIBIOTICS) / len(CORE_ANTIBIOTICS) * 100, 1)
    avg_f1 = round(sum(METRICS[ab]["f1"] for ab in CORE_ANTIBIOTICS) / len(CORE_ANTIBIOTICS), 3)

    return render_template("index.html",
                           antibiotics=SUPPORTED_ANTIBIOTICS,
                           core_antibiotics=CORE_ANTIBIOTICS,
                           antibiotic_count=len(SUPPORTED_ANTIBIOTICS),
                           antibiotic_info=ANTIBIOTIC_INFO,
                           metrics=METRICS,
                           avg_accuracy=avg_accuracy,
                           avg_f1=avg_f1,
                           feature_importance=FI["per_antibiotic"],
                           feature_cols=FEATURE_COLS)


@app.route("/genomic_features")
def genomic_features_page():
    return render_template("genomic_features.html", feature_cols=FEATURE_COLS)


@app.route("/model_info")
def model_info_page():
    avg_f1 = round(sum(METRICS[ab]["f1"] for ab in CORE_ANTIBIOTICS) / len(CORE_ANTIBIOTICS), 3)
    avg_accuracy = round(sum(METRICS[ab]["accuracy"] for ab in CORE_ANTIBIOTICS) / len(CORE_ANTIBIOTICS) * 100, 1)

    training_summary = {
        "algorithm": "MultiOutputClassifier(RandomForestClassifier)",
        "estimators": 200,
        "max_depth": 12,
        "dataset_size": 3000,
        "feature_count": len(FEATURE_COLS),
        "data_split": "80/20 train-test split",
        "source": "PATRIC / BV-BRC + online phenotype extension"
    }

    return render_template(
        "model_info.html",
        antibiotics=CORE_ANTIBIOTICS,
        supported_antibiotics=SUPPORTED_ANTIBIOTICS,
        extension_antibiotics=[ab for ab in SUPPORTED_ANTIBIOTICS if ab not in CORE_ANTIBIOTICS],
        antibiotic_info=ANTIBIOTIC_INFO,
        metrics=METRICS,
        avg_f1=avg_f1,
        avg_accuracy=avg_accuracy,
        training_summary=training_summary
    )

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data.get("features", {})

    # Build feature vector (fill missing with 0)
    X = np.array([[float(features.get(f, 0)) for f in FEATURE_COLS]])

    predictions = {}
    for i, ab in enumerate(CORE_ANTIBIOTICS):
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
            "info":       ANTIBIOTIC_INFO[ab],
            "derived":    False
        }

    predictions.update(_infer_extended_predictions(predictions))

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
        seed_features, explicit_features = _extract_seed_features_from_medical_text(medical_text)
        llm_warning = None

        prompt = f"""
You are extracting AMR genomic features for a resistance model.

Task:
1) Read the medical/genomics text below.
2) Use provided seed feature values when they reflect explicit clues from the report.
3) Infer missing feature values from the report context.
3) Return ONLY a JSON object with exactly these keys: {", ".join(FEATURE_COLS)}

Rules:
- Values must be numeric and in range [0, 1].
- If there is not enough evidence, use 0.
- Keep explicit report-derived seed values unchanged unless the report clearly contradicts them.
- Do not include explanations, markdown, or extra keys.

Seed features extracted from the report:
{json.dumps(seed_features, sort_keys=True)}

Medical data:
{medical_text}
"""

        try:
            api_data, used_strict_json_mode = _call_generate_content(api_key, model_name, prompt)
        except (urllib.error.URLError, TimeoutError, socket.timeout):
            # Continue with seed-only extraction if remote LLM call is temporarily unavailable.
            api_data = None
            used_strict_json_mode = False
            llm_warning = "LLM request timed out or was unreachable. Using report-derived seed features only."

        if api_data:
            candidates = api_data.get("candidates", [])
            if not candidates:
                raise ValueError("AI Studio returned no candidate response.")

            parts = candidates[0].get("content", {}).get("parts", [])
            llm_text = "\n".join(part.get("text", "") for part in parts if "text" in part).strip()
            if not llm_text:
                raise ValueError("AI Studio returned an empty response.")
        else:
            llm_text = ""

        parser_fallback = False
        try:
            extracted = _extract_json_from_llm_response(llm_text) if llm_text else {}
            if not isinstance(extracted, dict):
                raise ValueError("AI Studio response format is invalid. Expected a JSON object.")
        except (ValueError, json.JSONDecodeError):
            extracted = {}
            parser_fallback = True

        # Merge strategy:
        # 1) Explicit F-values in report
        # 2) LLM inferred values
        # 3) Heuristic seed values from narrative report clues
        merged_features = {}
        for feature in FEATURE_COLS:
            if feature in explicit_features:
                merged_features[feature] = explicit_features[feature]
                continue

            if feature in extracted:
                merged_features[feature] = extracted[feature]
                continue

            merged_features[feature] = seed_features.get(feature, 0)

        normalized_features = _normalize_feature_payload(merged_features)
        non_zero_count = sum(1 for val in normalized_features.values() if val > 0)

        return jsonify({
            "features": normalized_features,
            "model": f"models/{model_name}",
            "non_zero_features": non_zero_count,
            "json_mode": "strict" if used_strict_json_mode else "prompt_only",
            "seed_features_detected": len(seed_features),
            "explicit_features_detected": len(explicit_features),
            "parser_fallback": parser_fallback,
            "llm_warning": llm_warning
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
    return jsonify({
        "metrics": METRICS,
        "feature_importance": FI,
        "supported_antibiotics": SUPPORTED_ANTIBIOTICS,
        "core_antibiotics": CORE_ANTIBIOTICS,
        "extension_antibiotics": [ab for ab in SUPPORTED_ANTIBIOTICS if ab not in CORE_ANTIBIOTICS]
    })

@app.route("/random_sample")
def random_sample():
    """Return a random realistic genome sample."""
    df = pd.read_csv("data/amr_dataset.csv")
    row = df.sample(1).iloc[0]
    features = {f: float(row[f]) for f in FEATURE_COLS}
    actual = {ab: row[ab] for ab in CORE_ANTIBIOTICS}
    return jsonify({"features": features, "actual": actual})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
