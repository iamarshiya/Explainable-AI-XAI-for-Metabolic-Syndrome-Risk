import streamlit as st
from utils import apply_custom_css


st.set_page_config(
    page_title="MetaRisk XAI Intelligence Platform",
    page_icon="🧬",
    layout="wide"
)

apply_custom_css()

st.title("🧬 MetaRisk XAI Intelligence Platform")
st.markdown("### Production-Grade Metabolic Syndrome Risk Assessment & Explainable AI Framework")
st.markdown("---")

st.markdown("""
<div class="luxury-card">
    <h3>Welcome to the MetaRisk Intelligence Hub</h3>
    <p>This platform integrates multi-model machine learning benchmarks, SHAP explainability analyses, and generative AI clinical reports (Gemini powered) mapped across two dedicated operational portals:</p>
    <ul>
        <li><b>👤 Patient Portal:</b> Interactive vitals entry, personal risk evaluation, scenario testing (What-If sandbox), and downloadable wellness summaries.</li>
        <li><b>🩺 Clinician / Technical Portal:</b> Population cohort analytics, multi-model performance evaluation, and automated clinical SOAP notes.</li>
    </ul>
    <p><i>Please use the left sidebar navigation to select your desired portal mode.</i></p>
</div>
""", unsafe_allow_html=True)