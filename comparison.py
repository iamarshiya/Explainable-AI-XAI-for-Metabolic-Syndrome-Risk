import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
import shap

def prepare_clinical_data(df_raw):
    df_feat = df_raw.copy()
    
    # Handle implicit zeros via median imputation
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_cols:
        df_feat[col] = df_feat[col].replace(0, np.nan)
        df_feat[col] = df_feat[col].fillna(df_feat.groupby('Outcome')[col].transform('median'))
    
    eps = 1e-6
    # Composite Clinical Risk Features
    df_feat['Glucose_BMI_interaction'] = df_feat['Glucose'] * df_feat['BMI']
    df_feat['Insulin_Glucose_ratio'] = df_feat['Insulin'] / (df_feat['Glucose'] + eps)
    df_feat['Genetic_Age_exposure'] = df_feat['DiabetesPedigreeFunction'] * df_feat['Age']
    df_feat['BP_Age_ratio'] = df_feat['BloodPressure'] / (df_feat['Age'] + eps)
    
    return df_feat

def main():
    print("1. Preparing Data and Training Model...")
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    df = pd.read_csv(url)
    df_clean = prepare_clinical_data(df)

    X = df_clean.drop(columns=['Outcome'])
    y = df_clean['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_model.fit(X_train, y_train)

    print("\n2. Computing Feature Importance Metrics...")
    # Metric 1: Gini Importance (Model-centric)
    gini_imp = rf_model.feature_importances_

    # Metric 2: SHAP Importance (Prediction-centric)
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test)
    
    if isinstance(shap_values, list):
        shap_values_risk = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_values_risk = shap_values[:, :, 1]
    else:
        shap_values_risk = shap_values

    shap_imp = np.abs(shap_values_risk).mean(axis=0)

    # Metric 3: Mutual Information (Data-centric / Statistical)
    mi_imp = mutual_info_classif(X_train, y_train, random_state=42)

    print("\n3. Constructing Comparison Table...")
    comp_df = pd.DataFrame({
        'Feature': X.columns,
        'Gini_Importance': gini_imp,
        'SHAP_Importance': shap_imp,
        'Mutual_Information': mi_imp
    })

    # Normalize scores to 0.0 - 1.0 scale for direct comparison
    for col in ['Gini_Importance', 'SHAP_Importance', 'Mutual_Information']:
        comp_df[f'{col}_Norm'] = comp_df[col] / comp_df[col].max()

    comp_df = comp_df.sort_values(by='SHAP_Importance_Norm', ascending=False)
    print("\nNormalized Importance Comparison:\n")
    print(comp_df[['Feature', 'Gini_Importance_Norm', 'SHAP_Importance_Norm', 'Mutual_Information_Norm']].to_string(index=False))

    print("\n4. Generating Comparative Bar Chart Plot...")
    plt.figure(figsize=(12, 6))
    x_axis = np.arange(len(comp_df))
    width = 0.25

    plt.bar(x_axis - width, comp_df['Gini_Importance_Norm'], width=width, label='Gini Importance (Tree Split)', color='#1f77b4')
    plt.bar(x_axis, comp_df['SHAP_Importance_Norm'], width=width, label='SHAP Score (Impact on Prediction)', color='#ff7f0e')
    plt.bar(x_axis + width, comp_df['Mutual_Information_Norm'], width=width, label='Mutual Information (Statistical Risk)', color='#2ca02c')

    plt.xticks(x_axis, comp_df['Feature'], rotation=45, ha='right')
    plt.ylabel('Normalized Score (0.0 - 1.0)')
    plt.title('Feature Importance Metric Comparison: Gini vs. SHAP vs. Mutual Information')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plot_filename = "feature_importance_comparison.png"
    plt.savefig(plot_filename, dpi=300)
    print(f" -> Saved comparative plot as '{plot_filename}'")
    plt.show()

if __name__ == "__main__":
    main()