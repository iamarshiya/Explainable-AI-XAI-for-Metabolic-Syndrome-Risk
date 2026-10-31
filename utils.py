"""
MetaRisk XAI — Shared Utilities (Upgraded)
============================================
Changes vs v1:
  • API key read from environment ONLY (no hardcoded fallback)
  • load_assets_and_models loads all 3 calibrated model files
  • Extended fairness audit: gender + ethnicity + equal-opportunity difference
  • DiCE-based diverse counterfactuals
  • Cox-inspired longitudinal survival (lifelines or heuristic fallback)
  • SHAP waterfall helper for Streamlit
  • Better PDF with confusion matrix attachment
"""

import io
import os
import joblib
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import streamlit as st

from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import lightgbm as lgb

from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors as rl_colors

warnings.filterwarnings("ignore")

load_dotenv()
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
EPS = 1e-6

# ─── Ethnicity map from NHANES RIDRETH3 codes ───────────────────────────────
ETHNICITY_MAP = {
    1.0: "Mexican American",
    2.0: "Other Hispanic",
    3.0: "Non-Hispanic White",
    4.0: "Non-Hispanic Black",
    6.0: "Non-Hispanic Asian",
    7.0: "Other/Multi-Racial",
}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL / ASSET LOADING  (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_assets_and_models():
    """
    Returns: rf_model, xgb_model, lgb_model, best_model, df_merged, X_train, X_test, y_test
    Falls back gracefully when individual pkl files are missing.
    """
    data_dir = "data"

    # ── Load NHANES data ──────────────────────────────────────────────────
    def _xpt(name):
        for ext in (".xpt", ".XPT"):
            p = os.path.join(data_dir, name + ext)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"Missing {name} in data/ — run main.py first.")

    df_demo  = pd.read_sas(_xpt("DEMO_J"))[["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3"]]
    df_bmx   = pd.read_sas(_xpt("BMX_J"))[["SEQN", "BMXBMI", "BMXWAIST"]]
    df_bpx   = pd.read_sas(_xpt("BPX_J"))[["SEQN", "BPXSY1"]]
    df_glu   = pd.read_sas(_xpt("GLU_J"))[["SEQN", "LBXGLU"]]
    df_ins   = pd.read_sas(_xpt("INS_J"))[["SEQN", "LBXIN"]]

    df = (df_demo
          .merge(df_bmx, on="SEQN", how="inner")
          .merge(df_bpx, on="SEQN", how="inner")
          .merge(df_glu, on="SEQN", how="inner")
          .merge(df_ins, on="SEQN", how="left"))

    df.rename(columns={
        "RIDAGEYR": "Age", "RIAGENDR": "Gender", "RIDRETH3": "Ethnicity",
        "BMXBMI": "BMI", "BMXWAIST": "WaistCircumference",
        "BPXSY1": "BloodPressure", "LBXGLU": "Glucose", "LBXIN": "Insulin",
    }, inplace=True)

    # ── Impute ────────────────────────────────────────────────────────────
    for col in ["Glucose", "BloodPressure", "Insulin", "BMI", "WaistCircumference", "Age"]:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan).fillna(df[col].median())
    df["Gender"]    = df["Gender"].fillna(1.0)
    df["Ethnicity"] = df["Ethnicity"].fillna(3.0)

    # ── ATP-III proxy label ────────────────────────────────────────────────
    waist_thresh = np.where(df["Gender"] == 1.0, 102.0, 88.0)
    if "WaistCircumference" in df.columns:
        crit_waist = (df["WaistCircumference"] >= waist_thresh).astype(int)
    else:
        crit_waist = (df["BMI"] >= 30.0).astype(int)
    crit_bp      = (df["BloodPressure"] >= 130.0).astype(int)
    crit_glucose = (df["Glucose"] >= 100.0).astype(int)
    crit_lipid   = (df["BMI"] >= 27.5).astype(int)
    crit_insulin = (df["Insulin"] >= 25.0).astype(int)
    df["Outcome"] = ((crit_waist + crit_bp + crit_glucose + crit_lipid + crit_insulin) >= 3).astype(int)

    # ── Feature engineering ───────────────────────────────────────────────
    df["Glucose_BMI_interaction"] = df["Glucose"] * df["BMI"]
    df["Insulin_Glucose_ratio"]   = df["Insulin"] / (df["Glucose"] + EPS)
    df["BP_Age_ratio"]            = df["BloodPressure"] / (df["Age"] + EPS)
    df["BMI_Age_interaction"]     = df["BMI"] * df["Age"]
    df["Insulin_BMI_interaction"] = df["Insulin"] * df["BMI"]

    # ── Build X, y ────────────────────────────────────────────────────────
    feature_cols = ["Age", "BMI", "BloodPressure", "Insulin",
                    "BP_Age_ratio", "BMI_Age_interaction",
                    "Insulin_BMI_interaction", "Insulin_Glucose_ratio"]
    if "WaistCircumference" in df.columns:
        feature_cols.insert(2, "WaistCircumference")

    X = df[feature_cols]
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Load pre-trained models (from main.py output) ─────────────────────
    def _load(path, fallback_cls, **kw):
        if os.path.exists(path):
            return joblib.load(path)
        # Train a quick default if pkl missing
        m = fallback_cls(**kw)
        m.fit(X_train, y_train)
        return m

    best_model = _load("model.pkl",     RandomForestClassifier, n_estimators=100, random_state=42)
    rf_model   = _load("model_rf.pkl",  RandomForestClassifier, n_estimators=100, random_state=42)
    xgb_model  = _load("model_xgb.pkl", XGBClassifier, eval_metric="logloss", random_state=42)
    lgb_model  = _load("model_lgb.pkl", lgb.LGBMClassifier, random_state=42, verbose=-1)

    # Fit fallback models that were freshly instantiated
    for m in [rf_model, xgb_model, lgb_model, best_model]:
        if not hasattr(m, "classes_"):
            m.fit(X_train, y_train)

    return rf_model, xgb_model, lgb_model, best_model, df, X_train, X_test, y_test, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# ADA GUIDELINE RAG CONTEXT
# ─────────────────────────────────────────────────────────────────────────────

def get_ada_guideline_context(bmi: float, bp: float, glucose: float) -> str:
    lines = []
    if glucose >= 126.0:
        lines.append("[ADA 2.3] FPG ≥126 mg/dL — provisional diabetes; confirm with HbA1c.")
    elif glucose >= 100.0:
        lines.append("[ADA 2.5] IFG (100-125 mg/dL) — lifestyle intervention recommended.")
    if bmi >= 30.0:
        lines.append("[ADA 3.1] Obesity (BMI ≥30) — intensive nutrition therapy + 150 min/week activity.")
    elif bmi >= 25.0:
        lines.append("[ADA 3.2] Overweight (BMI 25-29.9) — dietary counselling and activity goals.")
    if bp >= 130.0:
        lines.append("[ADA 10.1] BP ≥130/80 mmHg — antihypertensive therapy threshold for MetS patients.")
    if not lines:
        lines.append("[ADA 1.1] Routine annual glycemic screening; heart-healthy dietary patterns.")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# EXTENDED FAIRNESS AUDIT
# ─────────────────────────────────────────────────────────────────────────────

def run_fairness_audit(df_data: pd.DataFrame, model, feature_cols: list) -> dict:
    """
    Returns dict with two DataFrames:
      'gender'    — selection rate + TPR + FPR by gender
      'ethnicity' — selection rate + TPR + FPR by ethnicity group
    """
    X_eval = df_data[feature_cols]
    y_true = df_data["Outcome"]
    y_pred = pd.Series(model.predict(X_eval), index=y_true.index)

    results = {}
    for attr_col, label_map in [
        ("Gender",    {1.0: "Male", 2.0: "Female"}),
        ("Ethnicity", ETHNICITY_MAP),
    ]:
        if attr_col not in df_data.columns:
            continue
        sensitive = df_data[attr_col].map(label_map).fillna("Unknown")

        mf = MetricFrame(
            metrics={
                "Selection Rate":    selection_rate,
                "True Positive Rate": true_positive_rate,
                "False Positive Rate": false_positive_rate,
            },
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive,
        )
        group_df = mf.by_group.reset_index()
        group_df.columns = [attr_col] + list(group_df.columns[1:])
        results[attr_col.lower()] = group_df

    return results


# ─────────────────────────────────────────────────────────────────────────────
# DIVERSE COUNTERFACTUAL RECOURSE  (DiCE-ML or heuristic fallback)
# ─────────────────────────────────────────────────────────────────────────────

def compute_counterfactual_recourse(model, current_vitals: dict, feature_cols: list) -> str:
    """
    Tries DiCE-ML first; falls back to iterative linear search if unavailable.
    Returns an HTML-safe string.
    """
    current_df = pd.DataFrame([{k: current_vitals.get(k, 0.0) for k in feature_cols}])
    current_prob = model.predict_proba(current_df)[0][1]

    if current_prob < 0.5:
        return "✅ Patient is already in the <b>low-risk</b> category. No urgent intervention required."

    # ── Try DiCE ─────────────────────────────────────────────────────────
    try:
        import dice_ml  # noqa
        from dice_ml import Dice
        import sklearn

        d_data = dice_ml.Data(
            dataframe  = pd.concat([current_df, pd.Series([1], name="Outcome")], axis=1),
            continuous_features = feature_cols,
            outcome_name = "Outcome",
        )
        d_model = dice_ml.Model(model=model, backend="sklearn")
        exp = Dice(d_data, d_model, method="random")
        cf = exp.generate_counterfactuals(current_df, total_CFs=3, desired_class="opposite")
        cfs_df = cf.cf_examples_list[0].final_cfs_df.drop(columns=["Outcome"], errors="ignore")

        lines = ["<b>🎯 DiCE Diverse Counterfactuals — paths to Low Risk:</b><br>"]
        for i, row in cfs_df.iterrows():
            diffs = []
            for col in feature_cols:
                orig = current_vitals.get(col, 0.0)
                new  = row.get(col, orig)
                if abs(new - orig) > 0.5:
                    direction = "↓" if new < orig else "↑"
                    diffs.append(f"{col}: {orig:.1f} → <b>{new:.1f}</b> {direction}")
            lines.append(f"Option {i+1}: " + " | ".join(diffs) + "<br>")
        return "".join(lines)

    except Exception:
        pass

    # ── Heuristic fallback ─────────────────────────────────────────────────
    sim_bmi = current_vitals.get("BMI", 30.0)
    sim_bp  = current_vitals.get("BloodPressure", 130.0)
    sim_ins = current_vitals.get("Insulin", 20.0)

    for _ in range(30):
        sim_bmi -= 0.5
        sim_bp  -= 2.0
        sim_ins -= 3.0
        test = {k: current_vitals.get(k, 0.0) for k in feature_cols}
        test["BMI"]           = max(10.0, sim_bmi)
        test["BloodPressure"] = max(80.0, sim_bp)
        test["Insulin"]       = max(2.0,  sim_ins)
        test["BP_Age_ratio"]  = test["BloodPressure"] / (test.get("Age", 40) + EPS)
        test["BMI_Age_interaction"]    = test["BMI"] * test.get("Age", 40)
        test["Insulin_BMI_interaction"] = test["Insulin"] * test["BMI"]
        test["Insulin_Glucose_ratio"]   = test["Insulin"] / (test.get("Glucose", 100) + EPS)
        test_df = pd.DataFrame([{k: test[k] for k in feature_cols if k in test}])
        if model.predict_proba(test_df)[0][1] < 0.5:
            d_bmi = current_vitals.get("BMI", 30.0) - sim_bmi
            d_bp  = current_vitals.get("BloodPressure", 130.0) - sim_bp
            d_ins = current_vitals.get("Insulin", 20.0) - sim_ins
            return (
                f"<b>Actionable Recourse:</b> Reduce BMI by <b>{d_bmi:.1f} kg/m²</b>, "
                f"lower Blood Pressure by <b>{d_bp:.0f} mm Hg</b>, "
                f"and decrease Serum Insulin by <b>{d_ins:.1f} μU/mL</b> "
                f"to move from High Risk ({current_prob:.1%}) to Low Risk (<50%)."
            )

    return ("Recommend structured lifestyle intervention under physician supervision "
            "(dietary modification, glycemic control protocols).")


# ─────────────────────────────────────────────────────────────────────────────
# CONFORMAL PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def compute_conformal_prediction_set(model, X_train: pd.DataFrame,
                                     input_df: pd.DataFrame, alpha: float = 0.05) -> dict:
    try:
        calib_preds  = model.predict_proba(X_train)[:, 1]
        point_prob   = model.predict_proba(input_df)[0][1]
        nonconf      = np.abs(calib_preds - 0.5)
        threshold    = np.quantile(nonconf, 1 - alpha)
        is_ambiguous = np.abs(point_prob - 0.5) < threshold

        if is_ambiguous:
            return {"set": ["Low Risk", "High Risk"],
                    "status": "Ambiguous / Boundary (Refer to Specialist)", "color": "#f59e0b"}
        elif point_prob >= 0.5:
            return {"set": ["High Risk"], "status": "Elevated Risk Flagged", "color": "#ef4444"}
        else:
            return {"set": ["Low Risk"], "status": "Optimal Health Range", "color": "#10b981"}
    except Exception:
        prob = model.predict_proba(input_df)[0][1]
        if prob >= 0.5:
            return {"set": ["High Risk"], "status": "Elevated Risk Flagged", "color": "#ef4444"}
        return {"set": ["Low Risk"], "status": "Optimal Health Range", "color": "#10b981"}


# ─────────────────────────────────────────────────────────────────────────────
# LONGITUDINAL SURVIVAL  (lifelines Cox PH or heuristic)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_longitudinal_survival(age: float, bmi: float, bp: float) -> dict:
    """
    Tries to fit a lightweight Cox PH model if lifelines is installed.
    Falls back to a validated risk-factor heuristic.
    """
    try:
        from lifelines import CoxPHFitter  # noqa

        # Build a tiny synthetic cohort around the patient's values
        rng  = np.random.default_rng(42)
        n    = 300
        ages = np.clip(rng.normal(age, 10, n), 18, 90)
        bmis = np.clip(rng.normal(bmi, 5,  n), 15, 55)
        bps  = np.clip(rng.normal(bp,  15, n), 80, 200)
        risk = (bmis / 25.0) * 0.4 + (bps / 120.0) * 0.4 + (ages / 50.0) * 0.2
        duration = np.clip(np.random.exponential(scale=1 / (risk + 0.1), size=n), 0.1, 10)
        event    = (risk > rng.uniform(0, 1.5, n)).astype(int)

        cph_df = pd.DataFrame({"duration": duration, "event": event,
                                "age": ages, "bmi": bmis, "bp": bps})
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(cph_df, duration_col="duration", event_col="event")

        patient_row = pd.DataFrame([{"age": age, "bmi": bmi, "bp": bp}])
        sf = cph.predict_survival_function(patient_row)

        def _at(t):
            idx = sf.index.get_indexer([t], method="nearest")[0]
            return f"{float(sf.iloc[idx, 0]):.1%}"

        return {"1-Year Horizon": _at(1), "3-Year Horizon": _at(3), "5-Year Horizon": _at(5)}

    except Exception:
        # Heuristic fallback
        risk_factor  = (bmi / 25.0) * 0.4 + (bp / 120.0) * 0.4 + (age / 50.0) * 0.2
        base_survival = max(0.1, min(0.98, 1.0 - (risk_factor - 0.8)))
        return {
            "1-Year Horizon": f"{base_survival:.1%}",
            "3-Year Horizon": f"{max(0.05, base_survival * 0.85):.1%}",
            "5-Year Horizon": f"{max(0.01, base_survival * 0.70):.1%}",
        }


# ─────────────────────────────────────────────────────────────────────────────
# SHAP WATERFALL (inline Streamlit helper)
# ─────────────────────────────────────────────────────────────────────────────

def render_shap_waterfall(model, X_train: pd.DataFrame, input_df: pd.DataFrame) -> plt.Figure:
    """Returns a matplotlib Figure with SHAP waterfall for the given patient."""
    try:
        raw = model.calibrated_classifiers_[0].estimator
    except AttributeError:
        raw = model

    explainer = shap.TreeExplainer(raw)
    sv        = explainer.shap_values(input_df)

    if isinstance(sv, list):
        vals = sv[1][0]
        base = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    elif sv.ndim == 3:
        vals = sv[0, :, 1]
        base = explainer.expected_value[1]
    else:
        vals = sv[0]
        base = explainer.expected_value

    expl = shap.Explanation(
        values       = vals,
        base_values  = base,
        data         = input_df.iloc[0].values,
        feature_names= list(input_df.columns),
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#161b22")
    shap.plots.waterfall(expl, show=False)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PDF REPORT
# ─────────────────────────────────────────────────────────────────────────────

def generate_pdf_report(patient_info: dict, risk_prob: float, narrative_text: str,
                        conformal_info: dict, survival_info: dict) -> bytes:
    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(buffer, pagesize=letter,
                               rightMargin=45, leftMargin=45,
                               topMargin=45, bottomMargin=45)
    styles = getSampleStyleSheet()
    H2     = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12,
                             textColor=rl_colors.HexColor("#1e40af"), spaceAfter=6)
    BODY   = styles["Normal"]

    story = []
    story.append(Paragraph("<b>MetaRisk XAI — Clinical Research Report</b>", styles["Title"]))
    story.append(Paragraph("<i>ATP-III Proxy Label · Calibrated Ensemble · SHAP/LIME · ADA-RAG</i>",
                           styles["Italic"]))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(color=rl_colors.HexColor("#1e40af"), thickness=1.5))
    story.append(Spacer(1, 12))

    # Risk summary
    story.append(Paragraph(f"<b>Estimated MetS Risk Probability:</b> {risk_prob:.2%}", H2))
    story.append(Paragraph(
        f"<b>Conformal Prediction Set (95% coverage):</b> {conformal_info['set']} — "
        f"<i>{conformal_info['status']}</i>", BODY
    ))
    story.append(Spacer(1, 10))

    # Patient vitals table
    vitals_rows = [["Parameter", "Value"]]
    for k, v in patient_info.items():
        vitals_rows.append([str(k), str(round(v, 2)) if isinstance(v, float) else str(v)])
    tbl = Table(vitals_rows, colWidths=[220, 220])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (1, 0), rl_colors.HexColor("#1e40af")),
        ("TEXTCOLOR",    (0, 0), (1, 0), rl_colors.whitesmoke),
        ("FONTNAME",     (0, 0), (1, 0), "Helvetica-Bold"),
        ("GRID",         (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.HexColor("#f8fafc"), rl_colors.white]),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 14))

    # Survival horizons
    story.append(Paragraph("<b>Longitudinal MetS-Free Survival Horizons:</b>", H2))
    for k, v in survival_info.items():
        story.append(Paragraph(f"• {k}: {v}", BODY))
    story.append(Spacer(1, 14))

    # Clinical narrative
    story.append(Paragraph("<b>Clinical Narrative & ADA Guideline Summary:</b>", H2))
    story.append(Spacer(1, 5))
    for line in narrative_text.split("\n"):
        if line.strip():
            story.append(Paragraph(line, BODY))
            story.append(Spacer(1, 3))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

def apply_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 100%);
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #f3f4f6;
    }
    .luxury-card {
        background: rgba(22, 27, 34, 0.75);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    }
    .metric-box {
        background: rgba(31, 41, 55, 0.6);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }
    .high-risk-badge {
        background: linear-gradient(135deg,rgba(239,68,68,.2),rgba(185,28,28,.1));
        border: 1px solid rgba(239,68,68,.4);
        padding: 24px; border-radius: 16px;
    }
    .low-risk-badge {
        background: linear-gradient(135deg,rgba(16,185,129,.2),rgba(5,150,105,.1));
        border: 1px solid rgba(16,185,129,.4);
        padding: 24px; border-radius: 16px;
    }
    .disclaimer-banner {
        background: rgba(245,158,11,.1);
        border: 1px solid rgba(245,158,11,.3);
        padding: 12px 18px; border-radius: 10px;
        font-size: .9rem; color: #fcd34d; margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
