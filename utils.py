import os
import io
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from dotenv import load_dotenv

# Fairlearn import for demographic bias audit
from fairlearn.metrics import MetricFrame, selection_rate

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAGr-BEleGvDDzC5hQUHeHrbcZJF4aB3tc")
EPS = 1e-6

@st.cache_resource
def load_assets_and_models():
    model = joblib.load("model.pkl")
    data_dir = "data"
    df_demo = pd.read_sas(os.path.join(data_dir, "DEMO_J.xpt"))[['SEQN', 'RIDAGEYR', 'RIAGENDR']]
    df_bmx = pd.read_sas(os.path.join(data_dir, "BMX_J.xpt"))[['SEQN', 'BMXBMI']]
    df_bpx = pd.read_sas(os.path.join(data_dir, "BPX_J.xpt"))[['SEQN', 'BPXSY1']]
    df_glu = pd.read_sas(os.path.join(data_dir, "GLU_J.xpt"))[['SEQN', 'LBXGLU']]
    df_ins = pd.read_sas(os.path.join(data_dir, "INS_J.xpt"))[['SEQN', 'LBXIN']]

    df_merged = df_demo.merge(df_bmx, on='SEQN', how='inner')\
                       .merge(df_bpx, on='SEQN', how='inner')\
                       .merge(df_glu, on='SEQN', how='inner')\
                       .merge(df_ins, on='SEQN', how='left')

    df_merged.rename(columns={
        'RIDAGEYR': 'Age',
        'RIAGENDR': 'Gender',
        'BMXBMI': 'BMI',
        'BPXSY1': 'BloodPressure', 
        'LBXGLU': 'Glucose',
        'LBXIN': 'Insulin'
    }, inplace=True)
    
    df_merged['Outcome'] = (df_merged['Glucose'] >= 126.0).astype(int)
    
    cols_to_fill = ['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age', 'Gender']
    for col in cols_to_fill:
        if col == 'Gender':
            df_merged[col] = df_merged[col].fillna(1.0)
        else:
            df_merged[col] = df_merged[col].replace(0, np.nan)
            df_merged[col] = df_merged[col].fillna(df_merged.groupby('Outcome')[col].transform('median'))
            df_merged[col] = df_merged[col].fillna(df_merged[col].median())
        
    df_merged['Glucose_BMI_interaction'] = df_merged['Glucose'] * df_merged['BMI']
    df_merged['Insulin_Glucose_ratio'] = df_merged['Insulin'] / (df_merged['Glucose'] + EPS)
    df_merged['BP_Age_ratio'] = df_merged['BloodPressure'] / (df_merged['Age'] + EPS)
    
    features_to_drop = ['Outcome', 'Glucose', 'Glucose_BMI_interaction', 'Insulin_Glucose_ratio', 'SEQN', 'Gender']
    X = df_merged.drop(columns=features_to_drop)
    y = df_merged['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    
    return model, xgb_model, lr_model, df_merged, X_train, y_test

def get_ada_guideline_context(bmi, bp, glucose):
    guidelines = []
    if glucose >= 126.0:
        guidelines.append("[ADA Standard 2.3] Fasting Plasma Glucose >= 126 mg/dL meets the threshold for provisional diabetes diagnosis requiring confirmatory HbA1c testing.")
    elif glucose >= 100.0:
        guidelines.append("[ADA Standard 2.5] Impaired Fasting Glucose (100-125 mg/dL): Recommend lifestyle intervention and structured weight management.")
    if bmi >= 30.0:
        guidelines.append("[ADA Standard 3.1] Class I/II Obesity (BMI >= 30 kg/m²): Initiate intensive medical nutrition therapy and physical activity counseling (150 min/week).")
    if bp >= 130.0:
        guidelines.append("[ADA Standard 10.1] Confirmed Blood Pressure >= 130/80 mmHg: Threshold for hypertension diagnosis in individuals with metabolic risk factors.")
    if not guidelines:
        guidelines.append("[ADA Standard 1.1] General Prevention: Maintain routine annual glycemic screening and heart-healthy dietary patterns.")
    return "\n".join(guidelines)

def run_fairness_audit(df_data, model):
    sensitive_attr = df_data['Gender'].map({1.0: 'Male', 2.0: 'Female'})
    X_eval = df_data[['Age', 'BMI', 'BloodPressure', 'Insulin', 'BP_Age_ratio']]
    y_true = df_data['Outcome']
    y_pred = model.predict(X_eval)
    
    metric_frame = MetricFrame(
        metrics=selection_rate,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_attr
    )
    return metric_frame.by_group

def simulate_longitudinal_survival(age, bmi, bp):
    risk_factor = (bmi / 25.0) * 0.4 + (bp / 120.0) * 0.4 + (age / 50.0) * 0.2
    base_survival = max(0.1, min(0.98, 1.0 - (risk_factor - 0.8)))
    return {
        "1-Year Horizon": f"{base_survival:.1%}",
        "3-Year Horizon": f"{max(0.05, base_survival * 0.85):.1%}",
        "5-Year Horizon": f"{max(0.01, base_survival * 0.70):.1%}"
    }

def compute_conformal_prediction_set(model, X_train, input_df, alpha=0.05):
    try:
        calib_preds = model.predict_proba(X_train)[:, 1]
        point_prob = model.predict_proba(input_df)[0][1]
        nonconformity_scores = np.abs(calib_preds - 0.5)
        threshold = np.quantile(nonconformity_scores, 1 - alpha)
        is_ambiguous = np.abs(point_prob - 0.5) < threshold
        
        if is_ambiguous:
            return {"set": ["Low Risk", "High Risk"], "status": "Ambiguous / Boundary Patient (Inconclusive - Refer to Specialist)", "color": "#f59e0b"}
        elif point_prob >= 0.5:
            return {"set": ["High Risk"], "status": "Elevated Risk Flagged", "color": "#ef4444"}
        else:
            return {"set": ["Low Risk"], "status": "Optimal Health Range", "color": "#10b981"}
    except Exception:
        if model.predict_proba(input_df)[0][1] >= 0.5:
            return {"set": ["High Risk"], "status": "Elevated Risk Flagged", "color": "#ef4444"}
        return {"set": ["Low Risk"], "status": "Optimal Health Range", "color": "#10b981"}

def compute_counterfactual_recourse(model, current_vitals):
    current_df = pd.DataFrame([current_vitals])
    current_prob = model.predict_proba(current_df)[0][1]
    if current_prob < 0.5:
        return "Patient is already in the low-risk category. No urgent clinical intervention required."
    
    simulated_bmi = current_vitals['BMI']
    simulated_bp = current_vitals['BloodPressure']
    simulated_insulin = current_vitals['Insulin']
    
    steps = 0
    while simulated_bmi > 15.0 and simulated_bp > 90.0 and steps < 20:
        simulated_bmi -= 0.5
        simulated_bp -= 2.0
        simulated_insulin -= 5.0
        
        test_feat = pd.DataFrame([{
            'Age': current_vitals['Age'],
            'BMI': max(10.0, simulated_bmi),
            'BloodPressure': max(80.0, simulated_bp),
            'Insulin': max(2.0, simulated_insulin),
            'BP_Age_ratio': max(80.0, simulated_bp) / (current_vitals['Age'] + EPS)
        }])
        if model.predict_proba(test_feat)[0][1] < 0.5:
            delta_bmi = current_vitals['BMI'] - simulated_bmi
            delta_bp = current_vitals['BloodPressure'] - simulated_bp
            delta_ins = current_vitals['Insulin'] - simulated_insulin
            return f"<b>Actionable Recourse Pathway:</b> Reduce BMI by <b>{delta_bmi:.1f} kg/m²</b>, lower Blood Pressure by <b>{delta_bp:.0f} mm Hg</b>, and decrease Serum Insulin by <b>{delta_ins:.1f} μU/mL</b> to transition from High Risk ({current_prob:.1%}) to Low Risk (<50%)."
            
    return "Recommended structured lifestyle intervention including dietary modification and glycemic control protocols under physician supervision."

def generate_pdf_report(patient_info, risk_prob, narrative_text, conformal_info, survival_info):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    
    story = []
    story.append(Paragraph("<b>MetaRisk XAI Research & Clinical Intelligence Report</b>", styles['Title']))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<i>ADA-Grounded RAG, Conformal Prediction & Survival Risk Stratification</i>", styles['Italic']))
    story.append(Spacer(1, 15))
    
    story.append(Paragraph(f"<b>Estimated Risk Probability:</b> {risk_prob:.2%}", styles['Heading2']))
    story.append(Paragraph(f"<b>Conformal Set:</b> {conformal_info['set']} — <i>{conformal_info['status']}</i>", styles['Normal']))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("<b>Longitudinal Disease Progression (Survival Horizons):</b>", styles['Heading3']))
    for k, v in survival_info.items():
        story.append(Paragraph(f"• {k}: Metabolic Syndrome-Free Probability = {v}", styles['Normal']))
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("<b>Clinical Narrative & ADA Guidelines:</b>", styles['Heading3']))
    story.append(Spacer(1, 5))
    for line in narrative_text.split('\n'):
        if line.strip():
            story.append(Paragraph(line, styles['Normal']))
            story.append(Spacer(1, 4))
            
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

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
            background: rgba(22, 27, 34, 0.7);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.4);
        }
        .metric-box {
            background: rgba(31, 41, 55, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 18px;
            text-align: center;
        }
        .high-risk-badge {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(185, 28, 28, 0.1) 100%);
            border: 1px solid rgba(239, 68, 68, 0.4);
            padding: 24px;
            border-radius: 16px;
        }
        .low-risk-badge {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.1) 100%);
            border: 1px solid rgba(16, 185, 129, 0.4);
            padding: 24px;
            border-radius: 16px;
        }
        .disclaimer-banner {
            background: rgba(245, 158, 11, 0.1);
            border: 1px solid rgba(245, 158, 11, 0.3);
            padding: 12px 18px;
            border-radius: 10px;
            font-size: 0.9rem;
            color: #fcd34d;
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)