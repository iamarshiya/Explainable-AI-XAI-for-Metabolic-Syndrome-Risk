import os
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

def fetch_nhanes_data():
    print("1. Loading NHANES Multi-Year Population Data from local 'data/' directory...")
    
    # Define paths to local files inside the data directory
    data_dir = "data"
    
    # Helper function to find file regardless of extension case (.xpt vs .XPT)
    def get_file_path(filename):
        lower_path = os.path.join(data_dir, f"{filename}.xpt")
        upper_path = os.path.join(data_dir, f"{filename}.XPT")
        if os.path.exists(lower_path):
            return lower_path
        elif os.path.exists(upper_path):
            return upper_path
        else:
            raise FileNotFoundError(f"Could not find {filename} in {data_dir}/ folder. Make sure the 5 files are downloaded and placed in the 'data' folder.")

    print(" -> Reading SAS XPT datasets...")
    df_demo = pd.read_sas(get_file_path("DEMO_J"))[['SEQN', 'RIDAGEYR']]
    df_bmx = pd.read_sas(get_file_path("BMX_J"))[['SEQN', 'BMXBMI']]
    df_bpx = pd.read_sas(get_file_path("BPX_J"))[['SEQN', 'BPXSY1']]
    df_glu = pd.read_sas(get_file_path("GLU_J"))[['SEQN', 'LBXGLU']]
    df_ins = pd.read_sas(get_file_path("INS_J"))[['SEQN', 'LBXIN']]

    print(" -> Merging patient records by Sequence Number (SEQN)...")
    df_merged = df_demo.merge(df_bmx, on='SEQN', how='inner')\
                       .merge(df_bpx, on='SEQN', how='inner')\
                       .merge(df_glu, on='SEQN', how='inner')\
                       .merge(df_ins, on='SEQN', how='left')

    # Rename columns to standard clinical features
    df_merged.rename(columns={
        'RIDAGEYR': 'Age',
        'BMXBMI': 'BMI',
        'BPXSY1': 'BloodPressure', 
        'LBXGLU': 'Glucose',
        'LBXIN': 'Insulin'
    }, inplace=True)
    
    # Target variable: 1 for High Risk/Diabetic (Fasting Glucose >= 126 mg/dL)
    df_merged['Outcome'] = (df_merged['Glucose'] >= 126.0).astype(int)
    
    # Drop identifier column
    df_merged.drop(columns=['SEQN'], inplace=True)
    
    return df_merged

def prepare_clinical_data(df_raw):
    df_feat = df_raw.copy()
    
    # Handle missing values
    cols_to_fill = ['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age']
    for col in cols_to_fill:
        df_feat[col] = df_feat[col].replace(0, np.nan)
        df_feat[col] = df_feat[col].fillna(df_feat.groupby('Outcome')[col].transform('median'))
        df_feat[col] = df_feat[col].fillna(df_feat[col].median())
    
    eps = 1e-6
    # Engineered Clinical Interaction Features
    df_feat['Glucose_BMI_interaction'] = df_feat['Glucose'] * df_feat['BMI']
    df_feat['Insulin_Glucose_ratio'] = df_feat['Insulin'] / (df_feat['Glucose'] + eps)
    df_feat['BP_Age_ratio'] = df_feat['BloodPressure'] / (df_feat['Age'] + eps)
    
    return df_feat

def main():
    df = fetch_nhanes_data()
    df_clean = prepare_clinical_data(df)

    print(f"\n✅ NHANES Dataset Loaded: {df_clean.shape[0]} patients, {df_clean.shape[1]} features.")

    # Split dataset
    print("\n2. Training Random Forest Model on NHANES Data...")
    X = df_clean.drop(columns=['Outcome', 'Glucose', 'Glucose_BMI_interaction', 'Insulin_Glucose_ratio'])
    y = df_clean['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, "model.pkl")
    print(" -> Saved updated model as 'model.pkl'!")

    # Evaluation
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
    plt.title("SHAP Summary Plot - NHANES Global Feature Importance")
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
    
    # Custom wrapper to suppress the Scikit-Learn warning
    def custom_predict_proba(x_array):
        # Convert LIME's raw numpy arrays back to a DataFrame with column names
        df_temp = pd.DataFrame(x_array, columns=X_train.columns)
        return rf_model.predict_proba(df_temp)
    
    # Pass the raw numpy array (.values) to LIME, and use the custom wrapper
    exp = lime_explainer.explain_instance(
        data_row=X_test.iloc[patient_idx].values,
        predict_fn=custom_predict_proba
    )
    exp.save_to_file('patient_lime_explanation.html')
    print(" -> Saved 'patient_lime_explanation.html'")
    
    print("\nPipeline Complete! Processed real clinical NHANES population records.")

if __name__ == "__main__":
    main()