import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import shap
import lime
import lime.lime_tabular
import requests  
import io        

def fetch_nhanes_data():
    print("Fetching NHANES Multi-Year Population Data (CDC)...")
    
    # CDC URLs for the 2017-2018 Cycle
    url_demo = "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO_J.XPT"  
    url_exam = "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BMX_J.XPT"   
    url_bp = "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BPX_J.XPT"     
    url_glu = "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/GLU_J.XPT"    
    url_ins = "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/INS_J.XPT"    

    def download_xpt(url):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Check if CDC returned an HTML page or error block instead of binary XPT
        if b'<html' in response.content.lower() or b'<!doctype' in response.content.lower():
            raise ValueError(f"CDC server blocked the request or returned an HTML error page for URL: {url}")
            
        return pd.read_sas(io.BytesIO(response.content), format='xport')

    try:
        print(" -> Downloading Demographics & Exams (this might take a minute)...")
        df_demo = download_xpt(url_demo)[['SEQN', 'RIDAGEYR']]
        df_bmx = download_xpt(url_exam)[['SEQN', 'BMXBMI']]
        df_bpx = download_xpt(url_bp)[['SEQN', 'BPXSY1']]

        print(" -> Downloading Clinical Labs...")
        df_glu = download_xpt(url_glu)[['SEQN', 'LBXGLU']]
        df_ins = download_xpt(url_ins)[['SEQN', 'LBXIN']]
    except Exception as e:
        print(f"\n[Error downloading from CDC]: {e}")
        print("\nFallback: Using local backup dataset to keep your pipeline moving.")
        url_fallback = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
        df_fallback = pd.read_csv(url_fallback)
        # Map fallback columns to match NHANES structure expected below
        df_fallback.rename(columns={
            'Age': 'Age',
            'BMI': 'BMI',
            'BloodPressure': 'BloodPressure',
            'Glucose': 'Glucose',
            'Insulin': 'Insulin',
            'Outcome': 'Outcome'
        }, inplace=True)
        # Ensure dummy SEQN if needed or just return fallback
        return df_fallback

    print(" -> Merging clinical patient records...")
    df_merged = df_demo.merge(df_bmx, on='SEQN', how='inner')\
                       .merge(df_bpx, on='SEQN', how='inner')\
                       .merge(df_glu, on='SEQN', how='inner')\
                       .merge(df_ins, on='SEQN', how='left')

    # Rename columns to match clinical terminology
    df_merged.rename(columns={
        'RIDAGEYR': 'Age',
        'BMXBMI': 'BMI',
        'BPXSY1': 'BloodPressure', 
        'LBXGLU': 'Glucose',
        'LBXIN': 'Insulin'
    }, inplace=True)
    
    # Target variable: 1 for Diabetic (Fasting Glucose >= 126), 0 otherwise
    df_merged['Outcome'] = (df_merged['Glucose'] >= 126.0).astype(int)
    
    # Drop SEQN as it's just an ID
    df_merged.drop(columns=['SEQN'], inplace=True)
    
    return df_merged

def prepare_clinical_data(df_raw):
    df_feat = df_raw.copy()
    
    cols_to_fill = [col for col in ['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age'] if col in df_feat.columns]
    for col in cols_to_fill:
        df_feat[col] = df_feat[col].replace(0, np.nan)
        if 'Outcome' in df_feat.columns:
            df_feat[col] = df_feat[col].fillna(df_feat.groupby('Outcome')[col].transform('median'))
        df_feat[col] = df_feat[col].fillna(df_feat[col].median())
    
    eps = 1e-6
    # Engineered Features
    if 'Glucose' in df_feat.columns and 'BMI' in df_feat.columns:
        df_feat['Glucose_BMI_interaction'] = df_feat['Glucose'] * df_feat['BMI']
    if 'Insulin' in df_feat.columns and 'Glucose' in df_feat.columns:
        df_feat['Insulin_Glucose_ratio'] = df_feat['Insulin'] / (df_feat['Glucose'] + eps)
    if 'BloodPressure' in df_feat.columns and 'Age' in df_feat.columns:
        df_feat['BP_Age_ratio'] = df_feat['BloodPressure'] / (df_feat['Age'] + eps)
    
    return df_feat

def main():
    print("1. Loading and preparing data...")
    df = fetch_nhanes_data()
    df_clean = prepare_clinical_data(df)

    print(f"\nDataset Ready: {df_clean.shape[0]} patients, {df_clean.shape[1]} features.")

    # Split data
    print("2. Splitting and Training Random Forest Model...")
    X = df_clean.drop(columns=['Outcome'])
    y = df_clean['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, "model.pkl")
    print(" -> Model saved successfully as model.pkl!")

    # Evaluate
    y_pred = rf_model.predict(X_test)
    print("\nModel Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    print("\n3. Generating SHAP Global Explanations...")
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test)
    
    if isinstance(shap_values, list):
        shap_values_risk = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_values_risk = shap_values[:, :, 1]
    else:
        shap_values_risk = shap_values

    plt.figure()
    plt.title("SHAP Summary Plot - Global Feature Importance")
    shap.summary_plot(shap_values_risk, X_test, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png")
    print(" -> Saved 'shap_summary.png'")
    plt.close()

    print("\n4. Generating Local Patient Explanations (Patient Index 5)...")
    patient_idx = 5
    patient_data = X_test.iloc[patient_idx]
    
    if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1:
        base_value = explainer.expected_value[1]
    else:
        base_value = explainer.expected_value

    shap_html = shap.plots.force(
        base_value, 
        shap_values_risk[patient_idx, :], 
        patient_data
    )
    shap.save_html("patient_shap_explanation.html", shap_html)
    print(" -> Saved 'patient_shap_explanation.html'")

    print("\n5. Generating LIME Explanation...")
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns,
        class_names=['Low Risk', 'High Risk'],
        mode='classification'
    )
    
    exp = lime_explainer.explain_instance(
        data_row=X_test.iloc[patient_idx].values,
        predict_fn=rf_model.predict_proba
    )
    exp.save_to_file('patient_lime_explanation.html')
    print(" -> Saved 'patient_lime_explanation.html'")
    
    print("\nProcess Complete! Check your project folder for the generated images and HTML files.")

if __name__ == "__main__":
    main()