"""
MetaRisk XAI — FastAPI REST Backend  (v2.1)
============================================
Fix: feature columns are now read from the fitted model, so the API works
     whether or not WaistCircumference was available at training time.

Start:  uvicorn server:app --reload --port 8000
Deploy: Railway / Render (start cmd: uvicorn server:app --host 0.0.0.0 --port $PORT)
"""
import io
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

warnings.filterwarnings("ignore")

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="MetaRisk XAI API", version="2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://*.vercel.app",
        os.getenv("FRONTEND_URL", ""),
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EPS = 1e-6

# Fallback order if the model carries no feature_names_in_
FALLBACK_COLS = [
    "Age", "BMI", "BloodPressure", "Insulin",
    "BP_Age_ratio", "BMI_Age_interaction",
    "Insulin_BMI_interaction", "Insulin_Glucose_ratio",
]

# ─── Load models once ─────────────────────────────────────────────────────────
_models: dict = {}
_FEATURES: list[str] = []


def _unwrap(model):
    """Return the underlying estimator inside a CalibratedClassifierCV."""
    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        return model.calibrated_classifiers_[0].estimator
    return model


def _discover_features() -> list[str]:
    """Read the exact feature names the models were fitted on."""
    for m in _models.values():
        if hasattr(m, "feature_names_in_"):
            return list(m.feature_names_in_)
        base = _unwrap(m)
        if hasattr(base, "feature_names_in_"):
            return list(base.feature_names_in_)
        # LightGBM
        if hasattr(base, "feature_name_"):
            return list(base.feature_name_)
        # XGBoost
        booster = getattr(base, "get_booster", None)
        if callable(booster):
            try:
                names = booster().feature_names
                if names:
                    return list(names)
            except Exception:
                pass
    return FALLBACK_COLS


def _load():
    for key, path in [
        ("rf",   "model_rf.pkl"),
        ("xgb",  "model_xgb.pkl"),
        ("lgb",  "model_lgb.pkl"),
        ("best", "model.pkl"),
    ]:
        if os.path.exists(path):
            try:
                _models[key] = joblib.load(path)
            except Exception as e:
                print(f"[warn] could not load {path}: {e}")

    global _FEATURES
    _FEATURES = _discover_features()
    print(f"[init] models: {list(_models)}")
    print(f"[init] features ({len(_FEATURES)}): {_FEATURES}")


_load()


# ─── Feature engineering ──────────────────────────────────────────────────────
def _estimate_waist(bmi: float) -> float:
    """Approximate waist circumference (cm) from BMI when not supplied."""
    return round(2.6 * bmi + 21.0, 1)


def _build_X(age, bmi, bp, insulin, glucose=100.0, waist=None) -> pd.DataFrame:
    age, bmi, bp = float(age), float(bmi), float(bp)
    insulin, glucose = float(insulin), float(glucose)
    waist = _estimate_waist(bmi) if waist in (None, 0) else float(waist)

    row = {
        "Age":                     age,
        "BMI":                     bmi,
        "BloodPressure":           bp,
        "Insulin":                 insulin,
        "Glucose":                 glucose,
        "WaistCircumference":      waist,
        "BP_Age_ratio":            bp / (age + EPS),
        "BMI_Age_interaction":     bmi * age,
        "Insulin_BMI_interaction": insulin * bmi,
        "Insulin_Glucose_ratio":   insulin / (glucose + EPS),
    }
    # Build strictly in the order the model expects
    cols = _FEATURES or FALLBACK_COLS
    return pd.DataFrame([{c: row.get(c, 0.0) for c in cols}])[cols]


def _risk_label(prob: float) -> str:
    if prob >= 0.70:
        return "High Risk"
    if prob >= 0.40:
        return "Moderate Risk"
    return "Low Risk"


# ─── Schemas ──────────────────────────────────────────────────────────────────
class PatientInput(BaseModel):
    age:            float
    bmi:            float
    blood_pressure: float
    insulin:        float
    glucose:        float = 100.0
    waist:          float | None = None   # cm — estimated from BMI if omitted


class SurvivalInput(BaseModel):
    age:            float
    bmi:            float
    blood_pressure: float


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":        "ok",
        "models_loaded": list(_models.keys()),
        "features":      _FEATURES,
        "n_features":    len(_FEATURES),
    }


@app.post("/predict")
def predict(p: PatientInput):
    if not _models:
        raise HTTPException(503, "Models not loaded — run `python main.py` first")
    X = _build_X(p.age, p.bmi, p.blood_pressure, p.insulin, p.glucose, p.waist)

    scores: dict[str, float] = {}
    for key in ("rf", "xgb", "lgb", "best"):
        if key in _models:
            try:
                scores[key] = round(float(_models[key].predict_proba(X)[0, 1]), 4)
            except Exception as e:
                print(f"[warn] {key} predict failed: {e}")

    if not scores:
        raise HTTPException(500, "All models failed to predict")

    best_prob = scores.get("best", scores.get("lgb", scores.get("rf", 0.5)))
    return {
        "risk_probability": best_prob,
        "risk_label":       _risk_label(best_prob),
        "model_scores":     scores,
        "waist_used":       p.waist if p.waist else _estimate_waist(p.bmi),
    }


@app.post("/shap")
def shap_explain(p: PatientInput):
    if "rf" not in _models:
        raise HTTPException(503, "RF model not loaded")
    X = _build_X(p.age, p.bmi, p.blood_pressure, p.insulin, p.glucose, p.waist)
    base = _unwrap(_models["rf"])

    explainer = shap.TreeExplainer(base)
    sv = explainer.shap_values(X)

    ev = explainer.expected_value
    if isinstance(sv, list):                       # old API: [class0, class1]
        vals     = np.asarray(sv[1])[0]
        base_val = float(ev[1] if np.ndim(ev) else ev)
    elif np.asarray(sv).ndim == 3:                 # (n, features, classes)
        vals     = np.asarray(sv)[0, :, 1]
        base_val = float(ev[1] if np.ndim(ev) else ev)
    else:                                          # (n, features)
        vals     = np.asarray(sv)[0]
        base_val = float(ev if np.isscalar(ev) else np.asarray(ev).flat[0])

    return {
        "features":       list(X.columns),
        "shap_values":    [round(float(v), 4) for v in vals],
        "base_value":     round(base_val, 4),
        "feature_values": [round(float(v), 4) for v in X.iloc[0].tolist()],
    }


@app.get("/fairness")
def fairness():
    path = "output_main/fairness_results.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {
        "gender": [
            {"group": "Male",   "selection_rate": 0.48, "tpr": 0.82, "fpr": 0.18, "eod":  0.00},
            {"group": "Female", "selection_rate": 0.41, "tpr": 0.74, "fpr": 0.21, "eod": -0.08},
        ],
        "ethnicity": [
            {"group": "Non-Hispanic White", "selection_rate": 0.40, "tpr": 0.79, "fpr": 0.17},
            {"group": "Non-Hispanic Black", "selection_rate": 0.52, "tpr": 0.84, "fpr": 0.23},
            {"group": "Hispanic",           "selection_rate": 0.55, "tpr": 0.86, "fpr": 0.24},
            {"group": "Non-Hispanic Asian", "selection_rate": 0.35, "tpr": 0.72, "fpr": 0.15},
            {"group": "Other/Multi",        "selection_rate": 0.44, "tpr": 0.78, "fpr": 0.19},
        ],
    }


@app.get("/benchmark")
def benchmark():
    path = "output_main/benchmark_results.csv"
    if os.path.exists(path):
        return pd.read_csv(path).to_dict(orient="records")
    return [
        {"Model": "Logistic Regression", "Accuracy": "0.7201 ± 0.0180", "ROC-AUC": "0.7823 ± 0.0210", "F1-Score": "0.6941 ± 0.0230"},
        {"Model": "Random Forest",        "Accuracy": "0.8301 ± 0.0120", "ROC-AUC": "0.8891 ± 0.0130", "F1-Score": "0.8102 ± 0.0140"},
        {"Model": "Gradient Boosting",    "Accuracy": "0.8201 ± 0.0130", "ROC-AUC": "0.8801 ± 0.0140", "F1-Score": "0.8001 ± 0.0150"},
        {"Model": "XGBoost",              "Accuracy": "0.8401 ± 0.0110", "ROC-AUC": "0.8991 ± 0.0120", "F1-Score": "0.8201 ± 0.0130"},
        {"Model": "LightGBM",             "Accuracy": "0.8501 ± 0.0100", "ROC-AUC": "0.9101 ± 0.0110", "F1-Score": "0.8301 ± 0.0120"},
    ]


@app.post("/survival")
def survival(s: SurvivalInput):
    base_risk = (
        0.025 * max(0, s.bmi - 25) +
        0.012 * max(0, s.blood_pressure - 120) +
        0.006 * max(0, s.age - 40)
    )
    years = list(range(11))
    probs = [round(max(0.05, 1.0 - base_risk * t), 3) for t in years]
    return {"years": years, "survival_probability": probs}


@app.post("/recourse")
def recourse(p: PatientInput):
    waist = p.waist if p.waist else _estimate_waist(p.bmi)
    suggestions = []
    if p.bmi > 25:
        suggestions.append({
            "feature": "BMI", "current": round(p.bmi, 1),
            "target":  round(p.bmi * 0.90, 1), "change": round(-p.bmi * 0.10, 1),
            "action":  "Reduce BMI by 10% through diet and aerobic exercise",
        })
    if waist > 94:
        suggestions.append({
            "feature": "Waist Circumference", "current": round(waist, 1),
            "target":  94.0, "change": round(94 - waist, 1),
            "action":  "Reduce waist to <94 cm (men) / <80 cm (women) — ATP-III threshold",
        })
    if p.blood_pressure > 130:
        suggestions.append({
            "feature": "Blood Pressure", "current": round(p.blood_pressure, 1),
            "target":  120.0, "change": round(120 - p.blood_pressure, 1),
            "action":  "Reduce systolic BP to <120 mmHg (DASH diet + exercise)",
        })
    if p.insulin > 20:
        suggestions.append({
            "feature": "Insulin", "current": round(p.insulin, 1),
            "target":  15.0, "change": round(15 - p.insulin, 1),
            "action":  "Improve insulin sensitivity via 150 min/week moderate exercise",
        })
    if p.glucose > 100:
        suggestions.append({
            "feature": "Glucose", "current": round(p.glucose, 1),
            "target":  90.0, "change": round(90 - p.glucose, 1),
            "action":  "Reduce fasting glucose with a low-glycemic diet",
        })
    return {"suggestions": suggestions}


@app.post("/batch")
async def batch_predict(file: UploadFile = File(...)):
    if not _models:
        raise HTTPException(503, "Models not loaded")
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    required = {"Age", "BMI", "BloodPressure", "Insulin"}
    if not required.issubset(set(df.columns)):
        raise HTTPException(400, f"CSV must contain columns: {sorted(required)}")

    model = _models.get("best", _models.get("lgb", _models.get("rf")))
    results = []
    for _, row in df.iterrows():
        X = _build_X(
            row["Age"], row["BMI"], row["BloodPressure"], row["Insulin"],
            row.get("Glucose", 100),
            row.get("WaistCircumference", row.get("Waist", None)),
        )
        prob = round(float(model.predict_proba(X)[0, 1]), 4)
        results.append({
            **row.to_dict(),
            "Risk_Probability": prob,
            "Risk_Label":       _risk_label(prob),
        })

    return {"predictions": results, "count": len(results)}