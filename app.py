import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Page Configuration
st.set_page_config(page_title="MetaRisk XAI", page_icon="🩺", layout="centered")

st.title("🩺 MetaRisk XAI Dashboard")
st.markdown("Clinical Explainable AI Diagnostic & Reporting System (Pure Python)")

# Load Model
@st.cache_resource
def load_model():
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    return None

model = load_model()

if model is None:
    st.error("Model file 'model.pkl' not found! Please run your training script first.")
else:
    st.sidebar.header("Patient Clinical Vitals")
    
    # Input Widgets
    pregnancies = st.sidebar.number_input("Pregnancies", min_value=0.0, max_value=20.0, value=1.0)
    glucose = st.sidebar.number_input("Plasma Glucose (mg/dL)", min_value=0.0, max_value=300.0, value=120.0)
    bp = st.sidebar.number_input("Blood Pressure (mm Hg)", min_value=0.0, max_value=200.0, value=70.0)
    skin = st.sidebar.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0)
    insulin = st.sidebar.number_input("Serum Insulin (mu U/ml)", min_value=0.0, max_value=900.0, value=85.0)
    bmi = st.sidebar.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=70.0, value=25.5)
    dpf = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.sidebar.number_input("Age (Years)", min_value=0.0, max_value=120.0, value=30.0)

    if st.button("Run AI Diagnostic", type="primary"):
        # Prepare Data DataFrame
        input_data = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': bp,
            'SkinThickness': skin,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age
        }
        df_input = pd.DataFrame([input_data])
        
        # Feature Engineering Pipeline
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
        
        # Predict
        prob = model.predict_proba(X_pred)[0][1]
        prediction = int(model.predict(X_pred)[0])
        
        # Display Results
        col1, col2 = st.columns(2)
        with col1:
            if prediction == 1:
                st.error(f"**Assessment:** High Risk (Diabetic)\n\n**Probability:** {prob*100:.2f}%")
            else:
                st.success(f"**Assessment:** Low Risk (Non-Diabetic)\n\n**Probability:** {prob*100:.2f}%")
                
        with col2:
            st.info("**Engineered Highlights:**\n"
                    f"- Glucose-BMI: {df_input['Glucose_BMI_interaction'].values[0]:.1f}\n"
                    f"- Insulin-Glucose: {df_input['Insulin_Glucose_ratio'].values[0]:.4f}")

        # PDF Generation Logic for Streamlit
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
            ('BACKGROUND', (0,0), (1,0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0,0), (1,0), colors.whitesmoke),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ]))
        story.append(t)
        doc.build(story)

        with open(pdf_filename, "rb") as pdf_file:
            st.download_button(
                label="📥 Download Clinical PDF Report",
                data=pdf_file,
                file_name="Clinical_Risk_Report.pdf",
                mime="application/pdf"
            )