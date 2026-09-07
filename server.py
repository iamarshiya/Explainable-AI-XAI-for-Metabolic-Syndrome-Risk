from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

app = FastAPI(title="Clinical XAI Metabolic Risk API")

# Enable CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for local testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
MODEL_PATH = "model.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

# Define Pydantic Input Schema matching patient vitals
class PatientVitals(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

@app.get("/")
def home():
    return {"status": "FastAPI XAI Clinical Server is Running"}

@app.post("/predict")
def predict_risk(data: PatientVitals):
    if model is None:
        raise HTTPException(status_code=500, detail="Model file not found. Run main.py to save model.pkl first.")
    
    # Convert input to DataFrame
    input_data = data.dict()
    df_input = pd.DataFrame([input_data])
    
    # Feature Engineering Pipeline (matching your training script)
    eps = 1e-6
    df_input['Glucose_BMI_interaction'] = df_input['Glucose'] * df_input['BMI']
    df_input['Insulin_Glucose_ratio'] = df_input['Insulin'] / (df_input['Glucose'] + eps)
    df_input['Genetic_Age_exposure'] = df_input['DiabetesPedigreeFunction'] * df_input['Age']
    df_input['BP_Age_ratio'] = df_input['BloodPressure'] / (df_input['Age'] + eps)
    
    # Ensure column order matches training data
    features = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
        'BMI', 'DiabetesPedigreeFunction', 'Age', 
        'Glucose_BMI_interaction', 'Insulin_Glucose_ratio', 
        'Genetic_Age_exposure', 'BP_Age_ratio'
    ]
    X_pred = df_input[features]
    
    # Predict Probability and Class
    prob = model.predict_proba(X_pred)[0][1] # Probability of High Risk
    prediction = int(model.predict(X_pred)[0])
    risk_label = "High Risk (Diabetic)" if prediction == 1 else "Low Risk (Non-Diabetic)"
    
    return {
        "prediction": prediction,
        "risk_label": risk_label,
        "probability": round(float(prob) * 100, 2),
        "engineered_features": {
            "Glucose_BMI_interaction": round(float(df_input['Glucose_BMI_interaction'].values[0]), 2),
            "Insulin_Glucose_ratio": round(float(df_input['Insulin_Glucose_ratio'].values[0]), 4),
            "Genetic_Age_exposure": round(float(df_input['Genetic_Age_exposure'].values[0]), 4),
            "BP_Age_ratio": round(float(df_input['BP_Age_ratio'].values[0]), 4)
        }
    }

@app.post("/generate-pdf")
def generate_pdf(data: PatientVitals):
    pdf_filename = "Clinical_Risk_Report.pdf"
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor('#1f77b4'), spaceAfter=15)
    story.append(Paragraph("Clinical Metabolic Risk & XAI Report", title_style))
    story.append(Paragraph("Automated Diagnostic Summary generated via Explainable AI Framework", styles['Normal']))
    story.append(Spacer(1, 15))

    # Patient Vitals Table Data
    vitals_data = [
        ["Clinical Parameter", "Recorded Value"],
        ["Pregnancies", str(data.Pregnancies)],
        ["Plasma Glucose", f"{data.Glucose} mg/dL"],
        ["Blood Pressure", f"{data.BloodPressure} mm Hg"],
        ["Skin Thickness", f"{data.SkinThickness} mm"],
        ["Serum Insulin", f"{data.Insulin} mu U/ml"],
        ["Body Mass Index (BMI)", str(data.BMI)],
        ["Diabetes Pedigree Function", str(data.DiabetesPedigreeFunction)],
        ["Age", str(data.Age)]
    ]

    t = Table(vitals_data, colWidths=[200, 200])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (1,0), colors.HexColor('#1f77b4')),
        ('TEXTCOLOR', (0,0), (1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
    ]))
    story.append(t)
    story.append(Spacer(1, 20))

    # Build PDF
    doc.build(story)
    
    return FileResponse(pdf_filename, media_type='application/pdf', filename="Clinical_Risk_Report.pdf")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)