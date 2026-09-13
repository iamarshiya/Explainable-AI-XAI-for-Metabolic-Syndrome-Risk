import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google import genai
from google.genai import types
from utils import load_assets_and_models, generate_pdf_report, apply_custom_css, GEMINI_API_KEY


st.set_page_config(page_title="Clinician Portal - MetaRisk XAI", page_icon="🩺", layout="wide")
apply_custom_css()

model, xgb_model, lr_model, df_data = load_assets_and_models()

st.sidebar.header("📁 Cohort Configuration")
patient_idx = st.sidebar.number_input("Select Patient Row Index", min_value=0, max_value=len(df_data)-1, value=5)

features_to_drop = ['Outcome', 'Glucose', 'Glucose_BMI_interaction', 'Insulin_Glucose_ratio', 'SEQN']
X_full = df_data.drop(columns=features_to_drop)
selected_patient_row = X_full.iloc[patient_idx]
actual_outcome = df_data.iloc[patient_idx]['Outcome']

st.title("🩺 Clinician Portal: Decision Support & Benchmark")
st.markdown("""
    <div class="disclaimer-banner">
        ⚠️ <b>Clinical Warning:</b> This decision support framework assists licensed healthcare providers with algorithmic risk stratification. Final diagnosis and treatment plans remain solely the clinical responsibility of the attending physician.
    </div>
""", unsafe_allow_html=True)
st.markdown("---")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(f'<div class="metric-box"><span style="color: #94a3b8;">Age</span><h2 style="color: #f3f4f6;">{int(selected_patient_row["Age"])}</h2></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-box"><span style="color: #94a3b8;">BMI</span><h2 style="color: #f3f4f6;">{round(selected_patient_row["BMI"], 1)}</h2></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-box"><span style="color: #94a3b8;">Blood Pressure</span><h2 style="color: #f3f4f6;">{int(selected_patient_row["BloodPressure"])}</h2></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-box"><span style="color: #94a3b8;">Insulin</span><h2 style="color: #f3f4f6;">{round(selected_patient_row["Insulin"], 1)}</h2></div>', unsafe_allow_html=True)
with col5:
    status_color = "#f87171" if actual_outcome == 1 else "#34d399"
    status_text = "High Risk" if actual_outcome == 1 else "Low Risk"
    st.markdown(f'<div class="metric-box"><span style="color: #94a3b8;">Clinical Status</span><h2 style="color: {status_color}; font-size: 1.3rem;">{status_text}</h2></div>', unsafe_allow_html=True)

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
        st.markdown("Patient telemetry benchmarked against NHANES population clusters.")

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
    if st.button("Generate Dual-Persona Clinical Narrative", type="primary"):
        with st.spinner("Analyzing telemetry and querying Gemini API..."):
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
                st.session_state['clinician_narrative'] = response.text
                st.session_state['clinician_risk'] = rf_proba
                st.session_state['clinician_vitals'] = selected_patient_row.to_dict()
            except Exception as e:
                st.error(f"Error connecting to Gemini API: {e}")
                
    if 'clinician_narrative' in st.session_state:
        st.markdown("---")
        st.markdown(st.session_state['clinician_narrative'])
        st.markdown("<br>", unsafe_allow_html=True)
        clinician_pdf = generate_pdf_report(st.session_state['clinician_vitals'], st.session_state['clinician_risk'], st.session_state['clinician_narrative'])
        st.download_button("📥 Download Formal SOAP Note PDF", data=clinician_pdf, file_name="MetaRisk_Clinical_SOAP_Report.pdf", mime="application/pdf")
    st.markdown('</div>', unsafe_allow_html=True)