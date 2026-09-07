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

def prepare_clinical_data(df_raw):
    df_feat = df_raw.copy()
    
    # Handle implicit zeros
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_cols:
        df_feat[col] = df_feat[col].replace(0, np.nan)
        df_feat[col] = df_feat[col].fillna(df_feat.groupby('Outcome')[col].transform('median'))
    
    eps = 1e-6
    # Engineered Features
    df_feat['Glucose_BMI_interaction'] = df_feat['Glucose'] * df_feat['BMI']
    df_feat['Insulin_Glucose_ratio'] = df_feat['Insulin'] / (df_feat['Glucose'] + eps)
    df_feat['Genetic_Age_exposure'] = df_feat['DiabetesPedigreeFunction'] * df_feat['Age']
    df_feat['BP_Age_ratio'] = df_feat['BloodPressure'] / (df_feat['Age'] + eps)
    
    return df_feat

def main():
    print("1. Loading and preparing data...")
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    df = pd.read_csv(url)
    df_clean = prepare_clinical_data(df)

    # Split data
    X = df_clean.drop(columns=['Outcome'])
    y = df_clean['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    print("\n2. Training Random Forest Model...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, "model.pkl")
    print("Model saved successfully as model.pkl!")

    # Evaluate
    y_pred = rf_model.predict(X_test)
    print("Model Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    print("\n3. Generating SHAP Global Explanations...")
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test)
    
    # SHAP returns a list [class_0, class_1] or 3D array for binary classification. We want Class 1 (High Risk)
    if isinstance(shap_values, list):
        shap_values_risk = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_values_risk = shap_values[:, :, 1]
    else:
        shap_values_risk = shap_values

    # Show SHAP Summary Plot
    plt.figure()
    plt.title("SHAP Summary Plot - Global Feature Importance")
    shap.summary_plot(shap_values_risk, X_test, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png")
    plt.show()

    print("\n4. Generating Local Patient Explanations (Patient Index 5)...")
    patient_idx = 5
    patient_data = X_test.iloc[patient_idx]
    
    # Safely extract base value for Class 1 (High Risk)
    if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1:
        base_value = explainer.expected_value[1]
    else:
        base_value = explainer.expected_value

    # Save SHAP Force Plot as HTML
    shap_html = shap.plots.force(
        base_value, 
        shap_values_risk[patient_idx, :], 
        patient_data
    )
    shap.save_html("patient_shap_explanation.html", shap_html)
    print(" -> Saved 'patient_shap_explanation.html'")

    # Save LIME Explanation as HTML
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
