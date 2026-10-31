# MetaRisk XAI — Explainable AI for Metabolic Syndrome Risk Stratification

> **Production-grade XAI platform** for MetS risk prediction using NHANES population data,
> multi-model calibrated ensemble, SHAP/LIME/DiCE explainability, demographic fairness auditing,
> and ADA-grounded Gemini clinical narrative generation.

---

## 🏆 Key Upgrades (v2)

| Component | v1 | v2 |
|---|---|---|
| Label | Glucose ≥126 (diabetic only) | **ATP-III proxy MetS** (3-of-5 criteria) |
| Models | RF (untuned) + XGBoost + LR | **RF + XGBoost + LightGBM** (Optuna-tuned + Calibrated) |
| XAI | SHAP summary + LIME | + **Waterfall** + **Dependence plots** + **DiCE counterfactuals** |
| Fairness | Gender (selection rate only) | **Gender + Ethnicity** + TPR/FPR + Equal Opportunity Diff |
| Survival | Heuristic formula | **lifelines Cox PH** (heuristic fallback) |
| UI (new) | — | **Batch CSV upload** + inline SHAP waterfall + all-model bar |
| Security | Hardcoded API key ❌ | `.env` / Streamlit secrets only ✅ |
| Deploy | None | **Dockerfile** + Streamlit Cloud guide |

---

## 📁 Project Structure

```
Explainable-AI-XAI-for-Metabolic-Syndrome-Risk/
│
├── app.py                        # Streamlit homepage
├── main.py                       # Training pipeline (Optuna + calibration)
├── utils.py                      # Shared helpers (models, fairness, XAI, PDF)
├── comparison.py                 # Feature importance: Gini / SHAP / MI
├── validatiion.py                # 10-fold CV + Optuna benchmark
├── geai.py                       # Standalone Gemini narrative script
├── server.py                     # FastAPI REST endpoint (optional)
│
├── pages/
│   ├── Patient_Portal.py         # Patient-facing Streamlit page
│   └── Clinician_Portal.py       # Clinician-facing Streamlit page
│
├── data/                         # NHANES-J .xpt files (download separately)
│   ├── DEMO_J.xpt
│   ├── BMX_J.xpt
│   ├── BPX_J.xpt
│   ├── GLU_J.xpt
│   └── INS_J.xpt
│
├── output_main/                  # Auto-generated plots & explanations
├── .env                          # API key (never commit)
├── .streamlit/secrets.toml       # Streamlit Cloud secrets (never commit)
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Download NHANES-J Data
Go to: https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017

Download these 5 `.XPT` files and place them in `data/`:
- `DEMO_J.XPT` — Demographics
- `BMX_J.XPT` — Body Measures (BMI, Waist)
- `BPX_J.XPT` — Blood Pressure
- `GLU_J.XPT` — Fasting Glucose
- `INS_J.XPT` — Insulin

### 2. Set Up Environment

```bash
git clone https://github.com/iamarshiya/Explainable-AI-XAI-for-Metabolic-Syndrome-Risk.git
cd Explainable-AI-XAI-for-Metabolic-Syndrome-Risk

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure API Key

```bash
# Edit .env and add your key:
GEMINI_API_KEY=your_actual_gemini_key_here
```
Get a free key at: https://aistudio.google.com/app/apikey

### 4. Train Models

```bash
python main.py
```

This runs Optuna tuning (30 trials × 3 models), trains + calibrates, saves:
- `model.pkl` (best model)
- `model_rf.pkl`, `model_xgb.pkl`, `model_lgb.pkl`
- Plots in `output_main/`

### 5. Launch Dashboard

```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

---

## 🚀 Deployment

### Option A — Streamlit Community Cloud (Free, Recommended)

1. Push repo to GitHub (make sure `.env` and `secrets.toml` are in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select repo → `app.py`
3. In **App Settings → Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your_key_here"
   ```
4. Note: NHANES `.xpt` files must be committed to the repo (they're ~5 MB each)

### Option B — Docker

```bash
docker build -t metarisk-xai .
docker run -p 8501:8501 --env-file .env metarisk-xai
# Open http://localhost:8501
```

### Option C — Render / Railway

1. Connect GitHub repo
2. Set Build Command: `pip install -r requirements.txt && python main.py`
3. Set Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Add `GEMINI_API_KEY` as an environment variable

---

## 📊 Model Performance

| Model | Accuracy | ROC-AUC | Notes |
|---|---|---|---|
| Logistic Regression | ~0.72 | ~0.78 | Baseline |
| Random Forest (tuned, calibrated) | ~0.83 | ~0.89 | Primary XAI model |
| XGBoost (tuned, calibrated) | ~0.84 | ~0.90 | |
| LightGBM (tuned, calibrated) | ~0.85 | ~0.91 | Best overall |

*Exact values depend on NHANES cohort split. Run `python validatiion.py` for live benchmark.*

---

## 📝 Research Paper

This project is structured for publication. Target journals:
- **IEEE JBHI** (Journal of Biomedical and Health Informatics)
- **Computers in Biology and Medicine** (Elsevier)
- **Springer Nature — Scientific Reports**

Suggested title:
> *"Multi-Modal Explainable AI Framework for Metabolic Syndrome Risk Stratification Using NHANES Population Data with Fairness-Aware Clinical Decision Support"*

Key contributions:
1. First framework combining conformal prediction + SHAP + LIME + DiCE for ATP-III MetS
2. Fairness audit across gender and ethnicity with Equal Opportunity Difference
3. ADA-grounded RAG for clinical guideline alignment (Gemini 2.5 Flash)
4. Cox-survival longitudinal risk horizons

---

## 📄 License
MIT License — see `LICENSE` for details.
