import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables securely from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAGr-BEleGvDDzC5hQUHeHrbcZJF4aB3tc")

# Page Configuration for High-End Medical Dashboard Theme
st.set_page_config(
    page_title="MetaRisk XAI Intelligence Platform",
    page_icon="🧬",
    layout="wide"
)

# Advanced Luxury Medical Tech CSS Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 100%);
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #f3f4f6;
    }
    
    /* Luxury Glassmorphic Cards */
    .luxury-card {
        background: rgba(22, 27, 34, 0.7);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
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
        backdrop-filter: blur(8px);
    }
    
    .high-risk-badge {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(185, 28, 28, 0.1) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(239, 68, 68, 0.15);
    }
    
    .low-risk-badge {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.1) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.15);
    }

    /* Custom Header Styling */
    h1, h2, h3 {
        letter-spacing: -0.025em;
        font-weight: 700;
    }
    
    /* Sidebar Polish */
    [data-testid="stSidebar"] {
        background-color: rgba(11, 15, 25, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    </style>
""", unsafe_allow_html=True)

# Load Dataset & Train Multi-Models for Benchmark Comparison
@st.cache_resource
def load_assets_and_models():
    model = joblib.load("model.pkl")
    data_dir = "data"
    df_demo = pd.read_sas(os.path.join(data_dir, "DEMO_J.xpt"))[['SEQN', 'RIDAGEYR']]
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
        'BMXBMI': 'BMI',
        'BPXSY1': 'BloodPressure', 
        'LBXGLU': 'Glucose',
        'LBXIN': 'Insulin'
    }, inplace=True)
    
    df_merged['Outcome'] = (df_merged['Glucose'] >= 126.0).astype(int)
    
    cols_to_fill = ['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age']
    for col in cols_to_fill:
        df_merged[col] = df_merged[col].replace(0, np.nan)
        df_merged[col] = df_merged[col].fillna(df_merged.groupby('Outcome')[col].transform('median'))
        df_merged[col] = df_merged[col].fillna(df_merged[col].median())
        
    eps = 1e-6
    df_merged['Glucose_BMI_interaction'] = df_merged['Glucose'] * df_merged['BMI']
    df_merged['Insulin_Glucose_ratio'] = df_merged['Insulin'] / (df_merged['Glucose'] + eps)
    df_merged['BP_Age_ratio'] = df_merged['BloodPressure'] / (df_merged['Age'] + eps)
    
    features_to_drop = ['Outcome', 'Glucose', 'Glucose_BMI_interaction', 'Insulin_Glucose_ratio', 'SEQN']
    X = df_merged.drop(columns=features_to_drop)
    y = df_merged['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    
    return model, xgb_model, lr_model, df_merged, X_train, y_test

model, xgb_model, lr_model, df_data, X_train, y_test = load_assets_and_models()

# --- GLOBAL SIDEBAR NAVIGATION ---
st.sidebar.title("🎛️ Navigation Hub")
page_selection = st.sidebar.radio("Select Portal Mode", ["👤 Patient Portal (ss1)", "🩺 Clinician / Technical Portal (ss2)"])
st.sidebar.markdown("---")

EPS = 1e-6

# =============================================================
# PAGE 1: PATIENT PORTAL (ss1)
# =============================================================
if page_selection == "👤 Patient Portal (ss1)":
    st.sidebar.header("⚙️ Patient Vitals Entry")
    st.sidebar.markdown("Adjust your clinical telemetry below:")

    age = st.sidebar.slider("Age (Years)", 18.0, 100.0, 36.0, 1.0)
    bmi = st.sidebar.slider("Body Mass Index (BMI)", 10.0, 60.0, 31.4, 0.1)
    blood_pressure = st.sidebar.slider("Blood Pressure (mm Hg)", 80.0, 200.0, 140.0, 1.0)
    glucose = st.sidebar.slider("Plasma Glucose (mg/dL)", 70.0, 300.0, 147.0, 1.0)
    insulin = st.sidebar.slider("Serum Insulin (mu U/mL)", 2.0, 300.0, 208.5, 0.5)

    input_features = pd.DataFrame([{
        'Age': age,
        'BMI': bmi,
        'BloodPressure': blood_pressure,
        'Insulin': insulin,
        'BP_Age_ratio': blood_pressure / (age + EPS)
    }])

    st.title("🧬 MetaRisk XAI Intelligence Platform")
    st.markdown("##### *Patient Wellness & Explainable Risk Diagnostics*")
    st.markdown("---")

    tab_diagnostic, tab_xai, tab_report = st.tabs(["📊 Diagnostic Results", "🔍 XAI Feature Contributions", "🤖 Generate Report"])
    risk_proba = model.predict_proba(input_features)[0][1]

    with tab_diagnostic:
        col_risk, col_summary = st.columns([1, 1], gap="medium")
        
        with col_risk:
            st.markdown("### Risk Assessment")
            if risk_proba > 0.5:
                st.markdown(f"""
                <div class="high-risk-badge">
                    <span style="color: #f87171; font-weight: 600; text-transform: uppercase; font-size: 0.85rem; letter-spacing: 0.05em;">Elevated Health Alert</span>
                    <h1 style="color: #ffffff; font-size: 3rem; margin: 8px 0;">{risk_proba:.1%}</h1>
                    <p style="color: #fca5a5; margin: 0; font-size: 0.95rem;">High Probability Factor for Metabolic Syndrome</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="low-risk-badge">
                    <span style="color: #34d399; font-weight: 600; text-transform: uppercase; font-size: 0.85rem; letter-spacing: 0.05em;">Optimal Health Range</span>
                    <h1 style="color: #ffffff; font-size: 3rem; margin: 8px 0;">{risk_proba:.1%}</h1>
                    <p style="color: #a7f3d0; margin: 0; font-size: 0.95rem;">Low Probability Factor for Metabolic Syndrome</p>
                </div>
                """, unsafe_allow_html=True)
                
        with col_summary:
            st.markdown(f"""
            <div class="luxury-card" style="margin-bottom: 0; height: 100%;">
                <h3 style="margin-top: 0; font-size: 1.2rem; color: #94a3b8;">Patient Summary Metrics</h3>
                <hr style="border-color: rgba(255,255,255,0.08); margin: 12px 0;">
                <ul style="list-style: none; padding: 0; line-height: 2.2rem; font-size: 1.05rem;">
                    <li>🎯 <b>Age:</b> {age:.1f} years</li>
                    <li>⚖️ <b>BMI:</b> {bmi:.1f}</li>
                    <li>🩸 <b>Plasma Glucose:</b> {glucose:.1f} mg/dL</li>
                    <li>💓 <b>Blood Pressure:</b> {blood_pressure:.1f} mm Hg</li>
                    <li>🧪 <b>Serum Insulin:</b> {insulin:.1f} μU/mL</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="luxury-card">', unsafe_allow_html=True)
        st.subheader("📈 Vitals vs. Normal Threshold Benchmarks")
        
        vitals_chart_data = pd.DataFrame({
            'Biomarker': ['BMI', 'Blood Pressure', 'Glucose', 'Insulin'],
            'Your Value': [bmi, blood_pressure, glucose, insulin],
            'Upper Normal Limit': [25.0, 120.0, 100.0, 25.0]
        }).set_index('Biomarker')
        
        st.bar_chart(vitals_chart_data)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_xai:
        st.markdown('<div class="luxury-card">', unsafe_allow_html=True)
        st.subheader("Explainable AI Feature Contributions")
        importance_df = pd.DataFrame({
            'Feature': ['Age', 'BMI', 'BloodPressure', 'Insulin', 'BP_Age_ratio'],
            'Value': [age, bmi, blood_pressure, insulin, blood_pressure / (age + EPS)]
        })
        st.dataframe(importance_df, use_container_width=True)
        
        st.markdown("#### Native SHAP Force / Impact Breakdown")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_features)
        
        s_vals = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
        
        shap_df = pd.DataFrame({
            'Feature': list(input_features.columns),
            'SHAP Value': np.array(s_vals).flatten()[:len(input_features.columns)]
        })
        
        fig, ax = plt.subplots(figsize=(8, 3))
        fig.patch.set_facecolor('#161b22')
        ax.set_facecolor('#0b0f19')
        
        sns.barplot(x='SHAP Value', y='Feature', data=shap_df, palette='coolwarm', ax=ax)
        
        ax.set_xlabel("SHAP Value (Impact on Model Output)", color='#94a3b8')
        ax.tick_params(colors='#94a3b8')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_report:
        st.markdown('<div class="luxury-card">', unsafe_allow_html=True)
        st.subheader("🤖 Generative Clinical Narrative Layer (Gemini-Powered)")
        st.markdown("Receive personalized wellness guidance and straightforward lifestyle recommendations.")
        if st.button("Generate Patient Guidance Report", type="primary"):
            with st.spinner("Synthesizing patient recommendations with Gemini AI..."):
                try:
                    client = genai.Client(api_key=GEMINI_API_KEY)
                    prompt = f"Provide empathetic, plain-language actionable health guidance for a patient with risk probability {risk_proba:.2%}, BMI {bmi}, and Blood Pressure {blood_pressure}."
                    response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
                    st.markdown("---")
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"Error connecting to Gemini API: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

# =============================================================
# PAGE 2: CLINICIAN / TECHNICAL PORTAL (ss2)
# =============================================================
elif page_selection == "🩺 Clinician / Technical Portal (ss2)":
    st.sidebar.header("📁 Cohort Configuration")
    patient_idx = st.sidebar.number_input("Select Patient Row Index", min_value=0, max_value=len(df_data)-1, value=5)

    features_to_drop = ['Outcome', 'Glucose', 'Glucose_BMI_interaction', 'Insulin_Glucose_ratio', 'SEQN']
    X_full = df_data.drop(columns=features_to_drop)
    selected_patient_row = X_full.iloc[patient_idx]
    actual_outcome = df_data.iloc[patient_idx]['Outcome']

    st.title("🧬 Explainable AI Clinical Decision Support System")
    st.markdown("##### *Advanced Cohort Telemetry & Multi-Model Research Benchmark*")
    st.markdown("---")

    # Metrics Grid Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f'<div class="metric-box"><span style="color: #94a3b8; font-size: 0.85rem;">Age</span><h2 style="margin: 5px 0 0 0; color: #f3f4f6;">{int(selected_patient_row["Age"])}</h2></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-box"><span style="color: #94a3b8; font-size: 0.85rem;">BMI</span><h2 style="margin: 5px 0 0 0; color: #f3f4f6;">{round(selected_patient_row["BMI"], 1)}</h2></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-box"><span style="color: #94a3b8; font-size: 0.85rem;">Blood Pressure</span><h2 style="margin: 5px 0 0 0; color: #f3f4f6;">{int(selected_patient_row["BloodPressure"])}</h2></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-box"><span style="color: #94a3b8; font-size: 0.85rem;">Insulin</span><h2 style="margin: 5px 0 0 0; color: #f3f4f6;">{round(selected_patient_row["Insulin"], 1)}</h2></div>', unsafe_allow_html=True)
    with col5:
        status_color = "#f87171" if actual_outcome == 1 else "#34d399"
        status_text = "High Risk" if actual_outcome == 1 else "Low Risk"
        st.markdown(f'<div class="metric-box"><span style="color: #94a3b8; font-size: 0.85rem;">Clinical Status</span><h2 style="margin: 5px 0 0 0; color: {status_color}; font-size: 1.3rem;">{status_text}</h2></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    clinician_tab_eval, clinician_tab_multimodel, clinician_tab_genai = st.tabs(["🎯 Diagnostic & Cohort Map", "🔬 Multi-Model Benchmark", "🤖 Generative AI Reports"])

    patient_df_reshaped = pd.DataFrame([selected_patient_row])
    rf_proba = model.predict_proba(patient_df_reshaped)[0][1]
    xgb_proba = xgb_model.predict_proba(patient_df_reshaped)[0][1]
    lr_proba = lr_model.predict_proba(patient_df_reshaped)[0][1]

    with clinician_tab_eval:
        st.markdown('<div class="luxury-card">', unsafe_allow_html=True)
        col_diag_text, col_diag_plot = st.columns([1, 1])
        with col_diag_text:
            st.subheader("Model Diagnostic Evaluation")
            if rf_proba > 0.5:
                st.error(f"**Random Forest High Risk Detected** — Probability: **{rf_proba:.2%}**")
            else:
                st.success(f"**Random Forest Low Risk Status** — Probability: **{rf_proba:.2%}**")
            st.markdown("Patient telemetry benchmarked against NHANES population clusters using primary ensemble classification.")

        with col_diag_plot:
            fig, ax = plt.subplots(figsize=(6, 3))
            fig.patch.set_facecolor('#161b22')
            ax.set_facecolor('#0b0f19')
            sample_cohort = df_data.sample(min(150, len(df_data)), random_state=42)
            sns.scatterplot(data=sample_cohort, x='BMI', y='BloodPressure', hue='Outcome', palette='coolwarm', alpha=0.7, ax=ax)
            ax.scatter([selected_patient_row['BMI']], [selected_patient_row['BloodPressure']], color='#fbbf24', s=180, marker='*', label='Selected Patient')
            ax.tick_params(colors='#94a3b8')
            ax.xaxis.label.set_color('#94a3b8')
            ax.yaxis.label.set_color('#94a3b8')
            ax.legend(loc='upper left', fontsize='x-small', facecolor='#161b22', labelcolor='white')
            for spine in ax.spines.values():
                spine.set_edgecolor('#30363d')
            st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with clinician_tab_multimodel:
        st.markdown('<div class="luxury-card">', unsafe_allow_html=True)
        st.subheader("Side-by-Side Classifier Benchmark Comparison")
        st.markdown("Evaluating risk probability outputs across multiple machine learning architectures for rigorous research validation:")
        
        comparison_df = pd.DataFrame({
            'Model Architecture': ['Random Forest (Primary)', 'XGBoost Classifier', 'Logistic Regression'],
            'Predicted Risk Score': [f"{rf_proba:.2%}", f"{xgb_proba:.2%}", f"{lr_proba:.2%}"],
            'Algorithm Type': ['Ensemble (Bagging)', 'Ensemble (Boosting)', 'Linear Decision Boundary']
        })
        st.dataframe(comparison_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with clinician_tab_genai:
        st.markdown('<div class="luxury-card">', unsafe_allow_html=True)
        st.subheader("Generative AI Clinical Reports (Gemini Powered)")
        st.markdown("Generate formal provider SOAP notes and structural safety metrics.")
        if st.button("Generate Dual-Persona Clinical Narrative", type="primary"):
            with st.spinner("Analyzing patient telemetry and querying Gemini API..."):
                try:
                    client = genai.Client(api_key=GEMINI_API_KEY)
                    telemetry_payload = {
                        "patient_id": int(df_data.iloc[patient_idx]['SEQN']) if 'SEQN' in df_data.columns else patient_idx,
                        "risk_probability": round(float(rf_proba), 2),
                        "patient_vitals": selected_patient_row.to_dict()
                    }
                    
                    prompt = f"""
                    You are an autonomous clinical decision support system adhering strictly to ADA standards. Analyze this telemetry:
                    {telemetry_payload}
                    
                    Provide:
                    1. DOCTOR VIEW: Structured SOAP Note (Subjective, Objective, Assessment, Plan) with clinical justification.
                    2. PATIENT VIEW: Empathetic, plain-language actionable guidance avoiding complex jargon.
                    """
                    
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction="You are a rigorous clinical AI assistant providing evidence-based medical summaries.",
                            temperature=0.2,
                        ),
                    )
                    st.markdown("---")
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"Error connecting to Gemini API: {e}")
        st.markdown('</div>', unsafe_allow_html=True)