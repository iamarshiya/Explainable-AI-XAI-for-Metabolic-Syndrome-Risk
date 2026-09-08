import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import base64

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# -----------------------------------------
# 1. Page Configuration & Custom CSS
# -----------------------------------------
st.set_page_config(page_title="MetaRisk XAI", page_icon="🧬", layout="wide")

# Injecting Custom CSS for a Glassmorphic, Modern UI
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(30, 41, 59, 0.7) !important;
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Fancy Button Styling */
    div.stButton > button {
        background: linear-gradient(135deg, #0d9488 0%, #14b8a6 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #14b8a6 0%, #38bdf8 100%);
        box-shadow: 0 4px 15px rgba(20, 184, 166, 0.4);
        transform: translateY(-1px);
    }
    
    /* Custom Glassmorphic Cards for Results */
    .glass-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    
    .highlight-text {
        color: #38bdf8;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------
# 2. Header & UI Layout
# -----------------------------------------
col_logo, col_title = st.columns([1, 8])
with col_title:
    st.title("🧬 MetaRisk XAI Dashboard")
    st.markdown("*Clinical Explainable AI Diagnostic & Reporting System*")
st.markdown("---")

# -----------------------------------------
# 3. Model Loading
# -----------------------------------------
@st.cache_resource
def load_model():
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    return None

model = load_model()

if model is None:
    st.error("⚠️ Model file 'model.pkl' not found! Please run your main.py training script first.")
    st.stop()

# -----------------------------------------
# 4. Sidebar: Patient Inputs
# -----------------------------------------
st.sidebar.header("📋 Patient Vitals Entry")
st.sidebar.markdown("Enter clinical parameters below:")

pregnancies = st.sidebar.number_input("Pregnancies", min_value=0.0, max_value=20.0, value=1.0)
glucose = st.sidebar.slider("Plasma Glucose (mg/dL)", 0.0, 300.0, 120.0)
bp = st.sidebar.slider("Blood Pressure (mm Hg)", 0.0, 200.0, 70.0)
skin = st.sidebar.slider("Skin Thickness (mm)", 0.0, 100.0, 20.0)
insulin = st.sidebar.slider("Serum Insulin (mu U/ml)", 0.0, 900.0, 85.0)
bmi = st.sidebar.slider("Body Mass Index (BMI)", 0.0, 70.0, 25.5)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.sidebar.slider("Age (Years)", 1.0, 120.0, 30.0)

# -----------------------------------------
# 5. Main Application Logic
# -----------------------------------------
# Layout Tabs
tab1, tab2, tab3 = st.tabs(["📊 Diagnostic Results", "🧠 XAI Feature Engineering", "📄 Generate Report"])

# Trigger Analysis
if st.sidebar.button("Run AI Diagnostic 🚀"):
    
    # 5a. Data Preparation & Engineering
    input_data = {
        'Pregnancies': pregnancies, 'Glucose': glucose, 'BloodPressure': bp,
        'SkinThickness': skin, 'Insulin': insulin, 'BMI': bmi,
        'DiabetesPedigreeFunction': dpf, 'Age': age
    }
    df_input = pd.DataFrame([input_data])
    
    eps = 1e-6
    df_input['Glucose_BMI_interaction'] = df_input['Glucose'] * df_input['BMI']
    df_input['Insulin_Glucose_ratio'] = df_input['Insulin'] / (df_input['Glucose'] + eps)
    df_input['Genetic_Age_exposure'] = df_input['DiabetesPedigreeFunction'] * df_input['Age']
    df_input['BP_Age_ratio'] = df_input['BloodPressure'] / (df_input['Age'] + eps)
    
    features = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
        'BMI', 'DiabetesPedigreeFunction', 'Age', 
        'Glucose_BMI_interaction', 'Insulin_Glucose_ratio', 
        'Genetic_Age_exposure', 'BP_Age_ratio'
    ]
    X_pred = df_input[features]
    
    # 5b. Inference
    prob = model.predict_proba(X_pred)[0][1]
    prediction = int(model.predict(X_pred)[0])
    
    # -----------------------------------------
    # Tab 1: Diagnostic Results
    # -----------------------------------------
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            if prediction == 1:
                st.markdown(f"""
                <div class="glass-card" style="border-left: 4px solid #ef4444;">
                    <h3 style="color: #f87171; margin-top: 0;">High Risk Detected</h3>
                    <h1 style="font-size: 3rem; margin: 10px 0;">{prob*100:.1f}%</h1>
                    <p style="color: #cbd5e1;">Probability of Metabolic Syndrome / Diabetes</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="glass-card" style="border-left: 4px solid #10b981;">
                    <h3 style="color: #34d399; margin-top: 0;">Low Risk Assessment</h3>
                    <h1 style="font-size: 3rem; margin: 10px 0;">{prob*100:.1f}%</h1>
                    <p style="color: #cbd5e1;">Probability of Metabolic Syndrome / Diabetes</p>
                </div>
                """, unsafe_allow_html=True)

        with col_res2:
            st.markdown(f"""
            <div class="glass-card">
                <h4 style="margin-top:0;">Patient Summary</h4>
                <p><b>Age:</b> {age} years</p>
                <p><b>BMI:</b> {bmi}</p>
                <p><b>Plasma Glucose:</b> {glucose} mg/dL</p>
                <p><b>Blood Pressure:</b> {bp} mm Hg</p>
            </div>
            """, unsafe_allow_html=True)

    # -----------------------------------------
    # Tab 2: Engineered Features
    # -----------------------------------------
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="glass-card">
            <h4>Automated Feature Interactions</h4>
            <p>The Random Forest model dynamically engineered the following clinical parameters to enhance interpretability:</p>
        </div>
        """, unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Glucose-BMI Interaction", f"{df_input['Glucose_BMI_interaction'].values[0]:.1f}")
        c2.metric("Insulin-Glucose Ratio", f"{df_input['Insulin_Glucose_ratio'].values[0]:.4f}")
        c3.metric("Genetic Age Exposure", f"{df_input['Genetic_Age_exposure'].values[0]:.2f}")

    # -----------------------------------------
    # Tab 3: Report Generation
    # -----------------------------------------
    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("Generating professional medical assessment document...")
        
        # ReportLab PDF Generation
        pdf_filename = "Clinical_Risk_Report.pdf"
        doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
        styles = getSampleStyleSheet()
        story = [
            Paragraph("Clinical Metabolic Risk & XAI Report", ParagraphStyle('Title', parent=styles['Heading1'], fontSize=16, textColor=colors.HexColor('#1f77b4'))),
            Spacer(1, 10),
            Paragraph(f"Assessment: {'High Risk' if prediction == 1 else 'Low Risk'} ({prob*100:.2f}% Probability)", styles['Normal']),
            Spacer(1, 15)
        ]
        
        vitals_table_data = [["Parameter", "Value"]] + [[k, str(v)] for k, v in input_data.items()]
        t = Table(vitals_table_data, colWidths=[200, 200])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (1,0), colors.HexColor('#0f172a')),
            ('TEXTCOLOR', (0,0), (1,0), colors.whitesmoke),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ]))
        story.append(t)
        doc.build(story)

        with open(pdf_filename, "rb") as pdf_file:
            st.download_button(
                label="📥 Download Secure Clinical PDF Report",
                data=pdf_file,
                file_name="Clinical_Risk_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
else:
    # Empty state UI
    st.info("👈 Enter patient vitals in the sidebar and click 'Run AI Diagnostic' to begin analysis.")