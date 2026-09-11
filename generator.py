import os
import json
from google import genai
from google.genai import types

def generate_clinical_narrative(patient_id, risk_probability, top_shap_features, vitals):
    # Construct structured JSON telemetry payload
    telemetry = {
        "patient_id": patient_id,
        "risk_probability": round(float(risk_probability), 2),
        "top_shap_features": top_shap_features,
        "patient_vitals": vitals
    }
    
    prompt = f"""
    You are an autonomous clinical decision support system. Analyze the following patient telemetry data:
    {json.dumps(telemetry, indent=2)}
    
    Provide a dual-persona response adhering strictly to American Diabetes Association (ADA) clinical standards:
    
    1. DOCTOR VIEW: Generate a structured SOAP Note (Subjective, Objective, Assessment, and Plan) complete with clinical justification and safety margins.
    2. PATIENT VIEW: Provide empathetic, plain-language actionable guidance avoiding dense medical jargon.
    
    Ensure zero hallucinations regarding dosages or medical contraindications.
    """

    print("Querying cloud LLM (Google Gemini API)...")
    
    # Initialize the client (picks up GEMINI_API_KEY from environment variables automatically)
    client = genai.Client()

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction="You are a rigorous clinical AI assistant providing evidence-based medical summaries.",
            temperature=0.2, # Low temperature for clinical accuracy and consistency
        ),
    )
    
    return response.text

if __name__ == "__main__":
    # Example telemetry payload for testing
    sample_vitals = {"Age": 52, "BMI": 31.4, "BloodPressure": 140, "Insulin": 18.5}
    sample_shap = {"BMI": "+0.31", "BloodPressure": "+0.22", "Age": "-0.04"}
    
    narrative_output = generate_clinical_narrative(
        patient_id="P-104",
        risk_probability=0.82,
        top_shap_features=sample_shap,
        vitals=sample_vitals
    )
    
    print("\n--- GENERATED CLINICAL NARRATIVE ---\n")
    print(narrative_output)
    
    # Save output to text file for clinical review
    with open("clinical_narrative_output.txt", "w", encoding="utf-8") as f:
        f.write(narrative_output)
    print("\n -> Saved output to 'clinical_narrative_output.txt'")