import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from google import genai
from utils import load_assets_and_models, generate_pdf_report, apply_custom_css, GEMINI_API_KEY, EPS


st.set_page_config(page_title="Patient Portal - MetaRisk XAI", page_icon="👤", layout="wide")
apply_custom_css()

model, xgb_model, lr_model, df_data = load_assets_and_models()

st.sidebar.header("⚙️ Patient Vitals Entry")
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

st.title("👤 Patient Portal: Wellness & Risk Assessment")
st.markdown("""
    <div class="disclaimer-banner">
        ⚠️ <b>Medical Disclaimer:</b> This tool provides automated risk estimation and educational guidance only. It does not replace professional medical advice, diagnosis, or treatment. Always consult a licensed physician regarding any health concerns.
    </div>
""", unsafe_allow_html=True)
st.markdown("---")

tab_diagnostic, tab_sandbox, tab_xai, tab_report = st.tabs(["📊 Diagnostic Results", "🧪 What-If Sandbox", "🔍 XAI Feature Contributions", "🤖 Generate Report"])
risk_proba = model.predict_proba(input_features)[0][1]

with tab_diagnostic:
    col_risk, col_summary = st.columns([1, 1], gap="medium")
    with col_risk:
        st.markdown("### Risk Assessment")
        if risk_proba > 0.5:
            st.markdown(f"""
            <div class="high-risk-badge">
                <span style="color: #f87171; font-weight: 600;">Elevated Health Alert</span>
                <h1 style="color: #ffffff; font-size: 3rem; margin: 8px 0;">{risk_proba:.1%}</h1>
                <p style="color: #fca5a5; margin: 0;">High Probability Factor for Metabolic Syndrome</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="low-risk-badge">
                <span style="color: #34d399; font-weight: 600;">Optimal Health Range</span>
                <h1 style="color: #ffffff; font-size: 3rem; margin: 8px 0;">{risk_proba:.1%}</h1>
                <p style="color: #a7f3d0; margin: 0;">Low Probability Factor for Metabolic Syndrome</p>
            </div>
            """, unsafe_allow_html=True)
            
    with col_summary:
        st.markdown(f"""
        <div class="luxury-card" style="margin-bottom: 0; height: 100%;">
            <h3 style="margin-top: 0; color: #94a3b8;">Patient Summary Metrics</h3>
            <hr style="border-color: rgba(255,255,255,0.08); margin: 12px 0;">
            <ul style="list-style: none; padding: 0; line-height: 2.2rem;">
                <li>🎯 <b>Age:</b> {age:.1f} years</li>
                <li>⚖️ <b>BMI:</b> {bmi:.1f}</li>
                <li>🩸 <b>Plasma Glucose:</b> {glucose:.1f} mg/dL</li>
                <li>💓 <b>Blood Pressure:</b> {blood_pressure:.1f} mm Hg</li>
                <li>🧪 <b>Serum Insulin:</b> {insulin:.1f} μU/mL</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

with tab_sandbox:
    st.markdown('<div class="luxury-card">', unsafe_allow_html=True)
    st.subheader("What-If Biomarker Scenario Sandbox")
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        sim_bmi = st.slider("Simulated BMI", 10.0, 60.0, float(bmi), 0.1, key="sim_bmi")
        sim_bp = st.slider("Simulated Blood Pressure (mm Hg)", 80.0, 200.0, float(blood_pressure), 1.0, key="sim_bp")
    with col_sb2:
        sim_insulin = st.slider("Simulated Insulin (μU/mL)", 2.0, 300.0, float(insulin), 0.5, key="sim_ins")
        
    sim_features = pd.DataFrame([{
        'Age': age, 'BMI': sim_bmi, 'BloodPressure': sim_bp, 'Insulin': sim_insulin, 'BP_Age_ratio': sim_bp / (age + EPS)
    }])
    sim_proba = model.predict_proba(sim_features)[0][1]
    delta_risk = sim_proba - risk_proba
    
    st.markdown("---")
    col_res1, col_res2 = st.columns(2)
    col_res1.metric("Baseline Risk Probability", f"{risk_proba:.1%}")
    # Change this line in pages/Patient_Portal.py:
    col_res2.metric("Simulated Risk Probability", f"{sim_proba:.1%}", delta=f"{delta_risk:.1%}", delta_color="inverse")
    st.markdown('</div>', unsafe_allow_html=True)

with tab_xai:
    st.markdown('<div class="luxury-card">', unsafe_allow_html=True)
    st.subheader("Explainable AI Feature Contributions")
    importance_df = pd.DataFrame({
        'Feature': ['Age', 'BMI', 'BloodPressure', 'Insulin', 'BP_Age_ratio'],
        'Value': [age, bmi, blood_pressure, insulin, blood_pressure / (age + EPS)]
    })
    st.dataframe(importance_df, use_container_width=True)
    
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
    if st.button("Generate Patient Guidance Report", type="primary"):
        with st.spinner("Synthesizing recommendations with Gemini AI..."):
            try:
                client = genai.Client(api_key=GEMINI_API_KEY)
                prompt = f"Provide empathetic, plain-language actionable health guidance for a patient with risk probability {risk_proba:.2%}, BMI {bmi}, and Blood Pressure {blood_pressure}."
                response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
                st.session_state['last_narrative'] = response.text
                st.session_state['last_risk'] = risk_proba
                st.session_state['last_vitals'] = {"Age": age, "BMI": bmi, "Blood Pressure": blood_pressure, "Glucose": glucose, "Insulin": insulin}
            except Exception as e:
                st.error(f"Error connecting to Gemini API: {e}")
                
    if 'last_narrative' in st.session_state:
        st.markdown("---")
        st.markdown(st.session_state['last_narrative'])
        st.markdown("<br>", unsafe_allow_html=True)
        pdf_bytes = generate_pdf_report(st.session_state['last_vitals'], st.session_state['last_risk'], st.session_state['last_narrative'])
        st.download_button("📥 Download Clinical PDF Report", data=pdf_bytes, file_name="MetaRisk_Patient_Report.pdf", mime="application/pdf")
    st.markdown('</div>', unsafe_allow_html=True)