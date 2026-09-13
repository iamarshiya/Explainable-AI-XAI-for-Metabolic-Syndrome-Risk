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
        
    df_merged['Glucose_BMI_interaction'] = df_merged['Glucose'] * df_merged['BMI']
    df_merged['Insulin_Glucose_ratio'] = df_merged['Insulin'] / (df_merged['Glucose'] + EPS)
    df_merged['BP_Age_ratio'] = df_merged['BloodPressure'] / (df_merged['Age'] + EPS)
    
    features_to_drop = ['Outcome', 'Glucose', 'Glucose_BMI_interaction', 'Insulin_Glucose_ratio', 'SEQN']
    X = df_merged.drop(columns=features_to_drop)
    y = df_merged['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    
    return model, xgb_model, lr_model, df_merged

def generate_pdf_report(patient_info, risk_prob, narrative_text):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    
    story = []
    story.append(Paragraph("<b>MetaRisk XAI Clinical Intelligence Report</b>", styles['Title']))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<i>Automated Decision Support & Risk Stratification Summary</i>", styles['Italic']))
    story.append(Spacer(1, 15))
    
    story.append(Paragraph(f"<b>Estimated Risk Probability:</b> {risk_prob:.2%}", styles['Heading2']))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("<b>Patient Telemetry Profile:</b>", styles['Heading3']))
    for k, v in patient_info.items():
        story.append(Paragraph(f"• {k}: {v}", styles['Normal']))
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("<b>Clinical Narrative & Recommendations:</b>", styles['Heading3']))
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

