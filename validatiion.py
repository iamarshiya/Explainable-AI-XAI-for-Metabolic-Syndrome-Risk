"""
MetaRisk XAI — Validation Pipeline (Upgraded)
===============================================
Upgrades:
  • LightGBM added to benchmark
  • Optuna replaces GridSearchCV for faster search
  • Reports mean ± 2*std confidence interval
  • Saves benchmark table to output_main/benchmark_results.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
os.makedirs("output_main", exist_ok=True)

EPS = 1e-6


def fetch_nhanes_data():
    data_dir = "data"

    def _p(name):
        for ext in (".xpt", ".XPT"):
            p = os.path.join(data_dir, name + ext)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"Missing {name} in data/")

    df_demo = pd.read_sas(_p("DEMO_J"))[["SEQN", "RIDAGEYR", "RIAGENDR"]]
    df_bmx  = pd.read_sas(_p("BMX_J"))[["SEQN", "BMXBMI"]]
    df_bpx  = pd.read_sas(_p("BPX_J"))[["SEQN", "BPXSY1"]]
    df_glu  = pd.read_sas(_p("GLU_J"))[["SEQN", "LBXGLU"]]
    df_ins  = pd.read_sas(_p("INS_J"))[["SEQN", "LBXIN"]]

    df = (df_demo
          .merge(df_bmx, on="SEQN", how="inner")
          .merge(df_bpx, on="SEQN", how="inner")
          .merge(df_glu, on="SEQN", how="inner")
          .merge(df_ins, on="SEQN", how="left"))

    df.rename(columns={"RIDAGEYR": "Age", "RIAGENDR": "Gender",
                       "BMXBMI": "BMI", "BPXSY1": "BloodPressure",
                       "LBXGLU": "Glucose", "LBXIN": "Insulin"}, inplace=True)

    for col in ["Glucose", "BloodPressure", "Insulin", "BMI", "Age"]:
        df[col] = df[col].replace(0, np.nan).fillna(df[col].median())

    crit_bp  = (df["BloodPressure"] >= 130).astype(int)
    crit_glu = (df["Glucose"] >= 100).astype(int)
    crit_bmi = (df["BMI"] >= 27.5).astype(int)
    crit_ins = (df["Insulin"] >= 25).astype(int)
    df["Outcome"] = ((crit_bp + crit_glu + crit_bmi + crit_ins) >= 2).astype(int)

    df["BP_Age_ratio"]            = df["BloodPressure"] / (df["Age"] + EPS)
    df["BMI_Age_interaction"]     = df["BMI"] * df["Age"]
    df["Insulin_BMI_interaction"] = df["Insulin"] * df["BMI"]
    df["Insulin_Glucose_ratio"]   = df["Insulin"] / (df["Glucose"] + EPS)

    return df


def main():
    print("=" * 60)
    print("  MetaRisk XAI — IEEE-Grade Validation Pipeline")
    print("=" * 60)

    df = fetch_nhanes_data()
    X  = df[["Age", "BMI", "BloodPressure", "Insulin",
              "BP_Age_ratio", "BMI_Age_interaction",
              "Insulin_BMI_interaction", "Insulin_Glucose_ratio"]]
    y  = df["Outcome"]

    print(f"\nDataset: {X.shape[0]} patients, {X.shape[1]} features")
    print(f"MetS prevalence: {y.mean():.2%}\n")

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # ─────────────────────────────────────────────────────────────
    # 1. 10-Fold CV Benchmark
    # ─────────────────────────────────────────────────────────────
    print("1. 10-Fold Stratified Cross-Validation Benchmark...")
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "Random Forest":    RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
        "XGBoost":          XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=42),
        "LightGBM":         lgb.LGBMClassifier(n_estimators=200, random_state=42, verbose=-1),
    }

    rows = []
    for name, model in models.items():
        auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        acc_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy",  n_jobs=-1)
        f1_scores  = cross_val_score(model, X, y, cv=cv, scoring="f1",        n_jobs=-1)
        rows.append({
            "Model":     name,
            "Accuracy":  f"{acc_scores.mean():.4f} ± {acc_scores.std()*2:.4f}",
            "ROC-AUC":   f"{auc_scores.mean():.4f} ± {auc_scores.std()*2:.4f}",
            "F1-Score":  f"{f1_scores.mean():.4f}  ± {f1_scores.std()*2:.4f}",
        })
        print(f"  {name:25s} | AUC {auc_scores.mean():.4f} | Acc {acc_scores.mean():.4f}")

    bench_df = pd.DataFrame(rows)
    bench_df.to_csv("output_main/benchmark_results.csv", index=False)
    print("\n  Saved output_main/benchmark_results.csv")
    print("\n" + bench_df.to_string(index=False))

    # ─────────────────────────────────────────────────────────────
    # 2. Optuna Hyperparameter Tuning (RF + LightGBM)
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("2. Optuna Hyperparameter Tuning (30 trials each)...")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                               stratify=y, random_state=42)

    # RF study
    def rf_obj(trial):
        m = RandomForestClassifier(
            n_estimators   = trial.suggest_int("n_est", 100, 500, step=100),
            max_depth       = trial.suggest_int("max_d", 3, 12),
            min_samples_split = trial.suggest_int("mss",  2, 20),
            min_samples_leaf  = trial.suggest_int("msl",  1, 10),
            random_state=42,
        )
        return cross_val_score(m, X_tr, y_tr, cv=5, scoring="roc_auc", n_jobs=-1).mean()

    rf_study = optuna.create_study(direction="maximize")
    rf_study.optimize(rf_obj, n_trials=30, show_progress_bar=False)
    best_rf_params = rf_study.best_params
    print(f"\n  Best RF params:       {best_rf_params}")
    print(f"  Best RF CV AUC:       {rf_study.best_value:.4f}")

    # LightGBM study
    def lgb_obj(trial):
        m = lgb.LGBMClassifier(
            n_estimators   = trial.suggest_int("n_est",  100, 500, step=100),
            max_depth       = trial.suggest_int("max_d",  3, 12),
            num_leaves      = trial.suggest_int("leaves", 20, 150),
            learning_rate   = trial.suggest_float("lr",   0.01, 0.3, log=True),
            min_child_samples = trial.suggest_int("mcs",  5, 50),
            random_state=42, verbose=-1,
        )
        return cross_val_score(m, X_tr, y_tr, cv=5, scoring="roc_auc", n_jobs=-1).mean()

    lgb_study = optuna.create_study(direction="maximize")
    lgb_study.optimize(lgb_obj, n_trials=30, show_progress_bar=False)
    best_lgb_params = lgb_study.best_params
    print(f"  Best LightGBM params: {best_lgb_params}")
    print(f"  Best LightGBM CV AUC: {lgb_study.best_value:.4f}")

    # ─────────────────────────────────────────────────────────────
    # 3. Calibrated final model evaluation
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("3. Final Calibrated Model Holdout Evaluation...")
    # Map short Optuna trial keys → real sklearn/lgb parameter names
    rf_mapped = {
        "n_estimators":     best_rf_params.get("n_est",  200),
        "max_depth":        best_rf_params.get("max_d",  6),
        "min_samples_split":best_rf_params.get("mss",    2),
        "min_samples_leaf": best_rf_params.get("msl",    1),
    }
    lgb_mapped = {
        "n_estimators":      best_lgb_params.get("n_est",  200),
        "max_depth":         best_lgb_params.get("max_d",  6),
        "num_leaves":        best_lgb_params.get("leaves", 31),
        "learning_rate":     best_lgb_params.get("lr",     0.1),
        "min_child_samples": best_lgb_params.get("mcs",    20),
    }
    best_rf   = CalibratedClassifierCV(
        RandomForestClassifier(**rf_mapped, random_state=42),
        method="isotonic", cv=5
    )
    best_lgbm = CalibratedClassifierCV(
        lgb.LGBMClassifier(**lgb_mapped, random_state=42, verbose=-1),
        method="isotonic", cv=5
    )
    best_rf.fit(X_tr, y_tr)
    best_lgbm.fit(X_tr, y_tr)

    for name, m in [("RF (calibrated)", best_rf), ("LightGBM (calibrated)", best_lgbm)]:
        acc = accuracy_score(y_te, m.predict(X_te))
        auc = roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])
        print(f"  {name:30s} | Accuracy: {acc:.4f} | AUC: {auc:.4f}")

    print("\n✅ Validation pipeline complete.")


if __name__ == "__main__":
    main()