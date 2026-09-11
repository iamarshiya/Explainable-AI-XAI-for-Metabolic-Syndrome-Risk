import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import os
from google import genai
from google.genai import types

# Page Configuration for Glassmorphic UI Theme
st.set_page_config(
    page_title="Metabolic Syndrome XAI & CDSS",
    page_icon="🧬",
    layout="wide"
)

# Load Trained Random Forest Model and Dataset
@st.cache_resource
def load_assets():
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
    
    return model, df_merged

model, df_data = load_assets()

# Sidebar: Patient Selection & API Configuration
st.sidebar.header("Configuration")
patient_idx = st.sidebar.number_input("Select Patient Row Index", min_value=0, max_value=len(df_data)-1, value=5)
api_key_input = st.sidebar.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))

features_to_drop = ['Outcome', 'Glucose', 'Glucose_BMI_interaction', 'Insulin_Glucose_ratio', 'SEQN']
X_full = df_data.drop(columns=features_to_drop)
selected_patient_row = X_full.iloc[patient_idx]
actual_outcome = df_data.iloc[patient_idx]['Outcome']

# Main Dashboard Header
st.title("🧬 Explainable AI Clinical Decision Support System")
st.markdown("### Metabolic Syndrome Risk Prediction & Generative Narrative Layer")
st.markdown("---")

# Display Vitals Metrics Row
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Age (Years)", int(selected_patient_row['Age']))
col2.metric("BMI", round(selected_patient_row['BMI'], 1))
col3.metric("Blood Pressure", int(selected_patient_row['BloodPressure']))
col4.metric("Insulin", round(selected_patient_row['Insulin'], 1))
col5.metric("Actual Status", "High Risk" if actual_outcome == 1 else "Low Risk")

st.markdown("---")

# Model Inference & Risk Scoring
patient_df_reshaped = pd.DataFrame([selected_patient_row])
risk_proba = model.predict_proba(patient_df_reshaped)[0][1]

st.subheader("🎯 Model Diagnostic Assessment")
if risk_proba > 0.5:
    st.error(f"**High Risk Detected** — Estimated Metabolic Syndrome / High Glucose Probability: **{risk_proba:.2%}**")
else:
    st.success(f"**Low Risk Status** — Estimated Metabolic Syndrome Probability: **{risk_proba:.2%}**")

st.markdown("---")

# Generative AI Narrative Section
st.subheader("🤖 Generative AI Clinical Reports (Gemini Powered)")
if st.button("Generate Dual-Persona Clinical Narrative"):
    if not api_key_input:
        st.warning("Please enter your Gemini API key in the sidebar configuration.")
    else:
        with st.spinner("Analyzing patient telemetry and querying Gemini API..."):
            try:
                client = genai.Client(api_key=api_key_input)
                
                top_features = selected_patient_row.to_dict()
                telemetry_payload = {
                    "patient_id": int(df_data.iloc[patient_idx]['SEQN']) if 'SEQN' in df_data.columns else patient_idx,
                    "risk_probability": round(float(risk_proba), 2),
                    "patient_vitals": top_features
                }
                
                prompt = f"""
                You are an autonomous clinical decision support system adhering strictly to American Diabetes Association (ADA) standards. Analyze this telemetry:
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
                
                st.markdown(response.text)
                
            except Exception as e:
                st.error(f"Error connecting to Gemini API: {e}")