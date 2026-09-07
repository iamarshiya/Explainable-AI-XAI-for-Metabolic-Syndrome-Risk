import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Ignore convergence warnings for clean output
warnings.filterwarnings("ignore")

def prepare_clinical_data(df_raw):
    df_feat = df_raw.copy()
    
    # Impute zero values
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
    print("--- IEEE Reviewer Validation Pipeline ---")
    print("Loading and preprocessing dataset...\n")
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    df = pd.read_csv(url)
    df_clean = prepare_clinical_data(df)

    X = df_clean.drop(columns=['Outcome'])
    y = df_clean['Outcome']
    
    # Stratified K-Fold ensures class balance in every fold
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # ---------------------------------------------------------
    # 1. Model Benchmarking & 10-Fold Cross Validation
    # ---------------------------------------------------------
    print("1. Running 10-Fold Cross-Validation & Benchmarking...")
    
    # Models to benchmark
    models = {
        "Logistic Regression (Baseline)": Pipeline([
            ('scaler', StandardScaler()), 
            ('lr', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Random Forest (Our XAI Model)": RandomForestClassifier(random_state=42),
        "Gradient Boosting (Advanced)": GradientBoostingClassifier(random_state=42)
    }

    benchmark_results = []
    
    for name, model in models.items():
        # Evaluate using ROC-AUC (better for imbalanced medical data than raw accuracy)
        auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        acc_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        
        benchmark_results.append({
            "Model": name,
            "Mean Accuracy": f"{acc_scores.mean():.4f} (+/- {acc_scores.std()*2:.4f})",
            "Mean ROC-AUC": f"{auc_scores.mean():.4f} (+/- {auc_scores.std()*2:.4f})"
        })

    benchmark_df = pd.DataFrame(benchmark_results)
    print(benchmark_df.to_string(index=False))
    print("\n" + "="*50 + "\n")

    # ---------------------------------------------------------
    # 2. Hyperparameter Tuning (GridSearch)
    # ---------------------------------------------------------
    print("2. Performing GridSearch for Random Forest Optimization...")
    
    # Split data for the final tuning phase
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Define the parameter grid to test
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [4, 6, 8, None],
        'min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5, # 5-fold internal validation
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print("\nOptimization Complete!")
    print(f"Best Parameters Found: {grid_search.best_params_}")
    print(f"Best Training ROC-AUC Score: {grid_search.best_score_:.4f}")
    
    # Test optimized model on holdout set
    best_rf = grid_search.best_estimator_
    test_acc = best_rf.score(X_test, y_test)
    print(f"Final Optimized Holdout Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()