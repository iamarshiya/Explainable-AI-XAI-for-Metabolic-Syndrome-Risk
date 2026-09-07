import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

# ---------------------------------------------------------
# 1. Load Dataset & Handle Implicit Zero Values
# ---------------------------------------------------------
# Download or load Pima Indians Diabetes Dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=columns)

# ---------------------------------------------------------
# 2. EDA: Original Features Subplot Grid Plot
# ---------------------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
fig.suptitle("Distribution of Original Features", fontsize=16)

orig_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
             'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

for i, col in enumerate(orig_cols):
    ax = axes[i // 3, i % 3]
    ax.hist(df[col], bins=25, edgecolor='none', color='#1f77b4')
    ax.set_title(col, fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("distribution_original_features.png", dpi=300)
plt.show()

# ---------------------------------------------------------
# 3. Feature Engineering Pipeline
# ---------------------------------------------------------
def engineer_diabetes_features(df_raw):
    df_feat = df_raw.copy()
    
    # Preprocessing: Replace biologically impossible zero values with NaN
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_cols:
        df_feat[col] = df_feat[col].replace(0, np.nan)
        # Median imputation grouped by target Outcome
        df_feat[col] = df_feat[col].fillna(df_feat.groupby('Outcome')[col].transform('median'))
    
    eps = 1e-6 # Epsilon guard against division by zero
    
    # Feature Engineering Logic
    df_feat['Insulin_Glucose_ratio'] = df_feat['Insulin'] / (df_feat['Glucose'] + eps)
    df_feat['Glucose_BMI_interaction'] = df_feat['Glucose'] * df_feat['BMI']
    df_feat['Pregnancy_Age_ratio'] = df_feat['Pregnancies'] / (df_feat['Age'] + eps)
    df_feat['SkinThickness_BMI_ratio'] = df_feat['SkinThickness'] / (df_feat['BMI'] + eps)
    df_feat['Genetic_Age_exposure'] = df_feat['DiabetesPedigreeFunction'] * df_feat['Age']
    df_feat['BP_Age_ratio'] = df_feat['BloodPressure'] / (df_feat['Age'] + eps)
    
    # BMI Risk Binning (Ordinally Encoded)
    bins = [0, 18.5, 24.9, 29.9, 100]
    labels = [0, 1, 2, 3] # Underweight, Normal, Overweight, Obese
    df_feat['BMI_Category'] = pd.cut(df_feat['BMI'], bins=bins, labels=labels).astype(int)
    
    return df_feat

df_engineered = engineer_diabetes_features(df)

# ---------------------------------------------------------
# 4. Engineered Features Subplot Grid Plot
# ---------------------------------------------------------
eng_cols = [
    'Insulin_Glucose_ratio', 'Glucose_BMI_interaction', 'Pregnancy_Age_ratio',
    'SkinThickness_BMI_ratio', 'Genetic_Age_exposure', 'BP_Age_ratio', 'BMI_Category'
]

fig, axes = plt.subplots(3, 3, figsize=(14, 10))
fig.suptitle("Distribution of Engineered Features", fontsize=16)

for i, col in enumerate(eng_cols):
    ax = axes[i // 3, i % 3]
    ax.hist(df_engineered[col], bins=25, edgecolor='none', color='#1f77b4')
    ax.set_title(col, fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)

# Hide extra unused subplot axes
axes[2, 1].axis('off')
axes[2, 2].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("distribution_engineered_features.png", dpi=300)
plt.show()

# ---------------------------------------------------------
# 5. Correlation Heatmap
# ---------------------------------------------------------
orig_feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
corr_matrix = pd.DataFrame(index=orig_feature_cols, columns=eng_cols)

for col_orig in orig_feature_cols:
    for col_eng in eng_cols:
        corr_matrix.loc[col_orig, col_eng] = df[col_orig].corr(df_engineered[col_eng])

corr_matrix = corr_matrix.astype(float)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1.0, vmax=1.0)
plt.title("Correlation Between Original and Engineered Features")
plt.xlabel("Engineered Features")
plt.ylabel("Original Features")
plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=300)
plt.show()

# ---------------------------------------------------------
# 6. Mutual Information Evaluation Plot
# ---------------------------------------------------------
X = df_engineered.drop(columns=['Outcome'])
y = df_engineered['Outcome']

mi_scores = mutual_info_classif(X, y, random_state=42)
mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=True)

plt.figure(figsize=(10, 6))
mi_series.plot(kind='barh', color='#1f77b4')
plt.title("Mutual Information Score by Feature")
plt.xlabel("Mutual Information Score")
plt.tight_layout()
plt.savefig("mutual_information_scores.png", dpi=300)
plt.show()