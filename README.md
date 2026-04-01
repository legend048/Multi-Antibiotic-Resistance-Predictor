# 🧬 AMR Multi-Antibiotic Resistance Predictor

> **AI-powered genomic resistance prediction for _E. coli_ — Ciprofloxacin, Ceftriaxone & Amoxicillin**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3%2B-black?style=flat-square)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?style=flat-square)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## 🔬 What This Does

This project is a **multi-task machine learning system** that predicts antibiotic resistance in *E. coli* genomes. Given 40 genomic features extracted from whole-genome sequencing data (gene presence/absence markers, mutation counts, and proportion scores), the core model predicts resistance outcomes for **three clinically important antibiotics**:

| Antibiotic | Drug Family | Treats |
|---|---|---|
| **Ciprofloxacin** | Fluoroquinolone | UTIs, respiratory infections |
| **Ceftriaxone** | 3rd-gen Cephalosporin | Pneumonia, meningitis, hospital infections |
| **Amoxicillin** | Penicillin | Ear infections, strep throat, chest infections |

Each prediction returns one of three standardised AST labels:

- 🟢 **S** — Susceptible (antibiotic kills the bacteria)
- 🟡 **I** — Intermediate (uncertain, dose/context dependent)
- 🔴 **R** — Resistant (antibiotic fails, bacteria survives)

The app also includes an **online phenotype extension model** (built from a public 7k-strain E. coli phenotype matrix) to infer additional antibiotics from core prediction patterns. Currently extended targets include Gentamicin, Ceftazidime, Piperacillin/Tazobactam, Meropenem, and Trimethoprim/Sulfamethoxazole.

---

## 📊 Model Performance

| Antibiotic | Accuracy | MCC | F1 (weighted) | AUC |
|---|---|---|---|---|
| Ciprofloxacin | 78.3% | 0.553 | 0.717 | **0.934** |
| Ceftriaxone | 83.5% | 0.632 | 0.783 | **0.935** |
| Amoxicillin | 79.8% | 0.572 | 0.732 | **0.934** |

**Algorithm:** `MultiOutputClassifier` wrapping `RandomForestClassifier` (200 trees, max depth 12)

---

## 🗄️ Dataset & Data Pipeline

### Source
Genomic data and Antimicrobial Susceptibility Testing (AST) results were sourced from the **PATRIC / BV-BRC** public database — the largest freely available bacterial genomics repository maintained by the U.S. Department of Energy.

- **Organism:** *Escherichia coli* (Gram-negative, clinically relevant)
- **Genomes collected:** 3,000 whole-genome sequencing (WGS) records
- **AST labels:** R / I / S for Ciprofloxacin, Ceftriaxone, and Amoxicillin

### Data Cleaning Steps

Raw data from PATRIC required significant cleaning before it was usable:

1. **Duplicate removal** — genomes with identical PATRIC IDs or redundant AST entries were dropped
2. **Missing label handling** — genomes missing AST results for any of the 3 antibiotics were excluded
3. **Outlier filtering** — genomes with abnormal feature distributions (>3 SD from mean) were flagged and reviewed
4. **Label standardisation** — raw MIC values and non-standard labels were mapped to S / I / R using EUCAST 2023 breakpoints
5. **Class balance check** — final label distribution confirmed: S ≈ 65%, I ≈ 15%, R ≈ 20%

### Feature Engineering

Features were extracted from cleaned genome assemblies using two bioinformatics tools:

**a) AMRFinderPlus** — identifies resistance genes:
```bash
amrfinderplus -i genome.fasta -o amr_genes.tsv
```

| Feature Group | Columns | What Was Extracted |
|---|---|---|
| F1–F15 | Binary (0/1) | Presence/absence of resistance genes & key mutations (e.g., *gyrA* S83L, *bla_TEM*, *bla_CTX-M-15*) |
| F16–F30 | Integer (0–10) | Counts of resistance elements, mobile genetic elements, efflux pump genes |
| F31–F40 | Real (0.00–1.00) | Proportional scores — fraction of genome with resistance markers, intact target sites |

**b) bcftools** — identifies point mutations:
```bash
bcftools mpileup -f reference.fasta genome.bam | bcftools call -mv -o variants.vcf
```
Mutations in *gyrA*, *parC*, *ompF*, and beta-lactamase regions were recorded as binary features.

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/Rishu-raj-02/AMR-Multi-Antibiotic-Resistance-Predictor.git
cd AMR-Multi-Antibiotic-Resistance-Predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python train_model.py
```
Reads `data/amr_dataset.csv`, trains the model, saves artifacts to `models/`.

### 4. Start the web app
```bash
python app.py
```
Open **http://localhost:5000** in your browser.

### 5. Use AI-assisted genomic autofill (optional)

Add this to `.env` in project root:

```env
ai_studio_key=YOUR_AI_STUDIO_KEY
```

1. Generate a Google AI Studio API key
2. Add it to `.env` as `ai_studio_key`
3. Upload medical/genomics notes (`.txt`, `.csv`, `.json`, `.md`, or `.pdf`)
4. Click **Extract & Fill Genomic Features**, then run prediction

The key in `.env` stays server-side and is not entered in the web UI.

You can upload plain narrative reports (without explicit `F1`-`F40` values). The backend now extracts marker clues from report text and combines them with LLM inference to autofill genomic features.

---

## 🗂️ Project Structure

```
AMR-Multi-Antibiotic-Resistance-Predictor/
│
├── data/
│   └── amr_dataset.csv          # Cleaned E. coli dataset (3,000 genomes × 44 cols)
│
├── models/                      # Auto-generated by train_model.py
│   ├── amr_model.pkl            # Trained MultiOutputClassifier
│   ├── encoders.pkl             # LabelEncoders for S/I/R
│   ├── metrics.json             # Per-antibiotic performance stats
│   ├── feature_importance.json  # Feature ranking per antibiotic
│   └── feature_cols.json        # Ordered feature column list
│
├── templates/
│   └── index.html               # Dark biotech dashboard UI
│
├── app.py                       # Flask web server + REST API
├── train_model.py               # Model training & evaluation script
├── requirements.txt
├── render.yaml                  # One-click Render.com deployment
├── Procfile                     # Gunicorn process config
└── README.md
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Interactive web dashboard |
| `POST` | `/predict` | Run core genomic prediction + extended online phenotype inference |
| `POST` | `/extract_features` | Use AI Studio + uploaded medical data to infer F1-F40 |
| `GET` | `/metrics` | Model performance stats |
| `GET` | `/random_sample` | Load a sample genome for demo |

### Example `/predict` request:
```json
POST /predict
{
  "features": {
    "F1": 1, "F2": 0, "F3": 1,
    "F4": 1, "F11": 1,
    "F16": 7, "F17": 3,
    "F31": 0.85, "F32": 0.20, "F33": 0.70
  }
}
```

### Example response:
```json
{
  "predictions": {
    "CIPRO": {
      "label": "R",
      "full": "Resistant",
      "confidence": 87.3,
      "probabilities": { "S": 0.07, "I": 0.06, "R": 0.87 }
    },
    "CEFTRIAXONE": { "label": "S", "confidence": 72.1 },
    "AMOXICILLIN": { "label": "R", "confidence": 81.5 }
  }
}
```

### Example `/extract_features` request:

`multipart/form-data`

- `api_key`: AI Studio API key
- `model`: Gemini model name (example: `gemini-2.0-flash`)
- `medical_file`: uploaded medical/genomics file

If a requested model is unavailable for your API key, the backend automatically selects a valid `generateContent` model.

### Example `/extract_features` response:
```json
{
  "features": {
    "F1": 1,
    "F2": 0,
    "F3": 0.4,
    "F4": 0,
    "F5": 0
  },
  "model": "models/gemini-2.0-flash",
  "non_zero_features": 9
}
```

---

## 🧬 Feature Reference

### F1–F15 · Binary (0 = absent, 1 = present)
Gene presence / mutation markers extracted via AMRFinderPlus:

| Feature | Biological Meaning | Linked Antibiotic |
|---|---|---|
| F1 | *gyrA* S83L mutation | Ciprofloxacin ↑R |
| F2 | *gyrA* D87G mutation | Ciprofloxacin ↑R |
| F3 | *bla_TEM* gene presence | Amoxicillin ↑R |
| F4 | Multi-drug resistance plasmid marker | All three ↑R |
| F7 | *bla_SHV* gene | Ceftriaxone ↑R |
| F11 | *bla_CTX-M-15* (ESBL) gene | Ceftriaxone ↑R |

### F16–F30 · Integer (0–10)
Count features — number of resistance mutations, mobile elements, efflux pump genes detected per genome assembly.

### F31–F40 · Real (0.00–1.00)
Proportion scores — fraction of genome containing resistance markers, proportion of intact drug-binding sites, etc.

---

## 🏥 Top Resistance Drivers (Feature Importance)

| Rank | Feature | Importance | Biological Role |
|---|---|---|---|
| 1 | F3 | 0.1018 | *bla_TEM* — Amoxicillin resistance |
| 2 | F2 | 0.0719 | *gyrA* mutation — Cipro resistance |
| 3 | F4 | 0.0570 | Multi-drug plasmid marker |
| 4 | F7 | 0.0553 | *bla_SHV* — Ceftriaxone resistance |
| 5 | F1 | 0.0536 | *gyrA* S83L — Cipro resistance |

---

## 🏥 Clinical Decision Support

The web app generates real-time clinical guidance:
- Flags **Resistant** predictions with evidence-based alternative drug suggestions
- Recommends confirmatory MIC lab testing for borderline (I) results
- Displays probability distributions across all three resistance classes

> ⚠️ **Disclaimer:** This tool is intended for research and educational purposes. Clinical treatment decisions must always be confirmed with certified laboratory antimicrobial susceptibility testing.

---

## 📖 How the Model Was Built

1. **Data Collection** — 3,000 *E. coli* WGS records with AST results from PATRIC/BV-BRC
2. **Cleaning** — duplicate removal, missing label exclusion, EUCAST breakpoint standardisation
3. **Feature Extraction** — AMRFinderPlus (resistance genes) + bcftools (point mutations)
4. **Feature Engineering** — 40 columns: 15 binary + 15 integer + 10 real-valued proportions
5. **Modelling** — MultiOutputClassifier (RandomForest, 200 estimators, depth 12)
6. **Evaluation** — 80/20 stratified split; Accuracy, MCC, F1-weighted, AUC-ROC (OvR)

---

## 🤝 Contributing

Pull requests welcome. For major changes, please open an issue first.

---

## 📄 License

MIT — see [LICENSE](LICENSE)

---

*Built for IIT Mandi Hackathon — E. coli · Ciprofloxacin · Ceftriaxone · Amoxicillin*
