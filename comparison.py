"""
MetaRisk XAI — Feature Importance Comparison (Upgraded)
=========================================================
Models: Random Forest, XGBoost, LightGBM
Metrics: Gini, SHAP, Mutual Information
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import lightgbm as lgb

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
os.makedirs("output_main", exist_ok=True)

EPS = 1e-6


def fetch_nhanes_data():
    data_dir = "data"

    def _path(name):
        for ext in (".xpt", ".XPT"):
            p = os.path.join(data_dir, name + ext)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"Missing {name} in data/")

    df_demo = pd.read_sas(_path("DEMO_J"))[["SEQN", "RIDAGEYR", "RIAGENDR"]]
    df_bmx  = pd.read_sas(_path("BMX_J"))[["SEQN", "BMXBMI"]]
    df_bpx  = pd.read_sas(_path("BPX_J"))[["SEQN", "BPXSY1"]]
    df_glu  = pd.read_sas(_path("GLU_J"))[["SEQN", "LBXGLU"]]
    df_ins  = pd.read_sas(_path("INS_J"))[["SEQN", "LBXIN"]]

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

    # ATP-III proxy label
    crit_bp  = (df["BloodPressure"] >= 130).astype(int)
    crit_glu = (df["Glucose"] >= 100).astype(int)
    crit_bmi = (df["BMI"] >= 27.5).astype(int)
    crit_ins = (df["Insulin"] >= 25).astype(int)
    df["Outcome"] = ((crit_bp + crit_glu + crit_bmi + crit_ins) >= 2).astype(int)

    df["Glucose_BMI_interaction"] = df["Glucose"] * df["BMI"]
    df["Insulin_Glucose_ratio"]   = df["Insulin"] / (df["Glucose"] + EPS)
    df["BP_Age_ratio"]            = df["BloodPressure"] / (df["Age"] + EPS)
    df["BMI_Age_interaction"]     = df["BMI"] * df["Age"]
    df["Insulin_BMI_interaction"] = df["Insulin"] * df["BMI"]

    df.drop(columns=["SEQN", "Gender", "Glucose"], inplace=True, errors="ignore")
    return df


def main():
    print("=== Feature Importance Comparison: Gini vs SHAP vs Mutual Information ===\n")

    df = fetch_nhanes_data()
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2,
                                                    stratify=y, random_state=42)

    # ── Train all 3 models ────────────────────────────────────────────────────
    rf  = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    xgb = XGBClassifier(n_estimators=200, max_depth=5, eval_metric="logloss", random_state=42)
    lgbm = lgb.LGBMClassifier(n_estimators=200, max_depth=5, random_state=42, verbose=-1)

    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)
    lgbm.fit(X_train, y_train)

    # ── SHAP for each model ───────────────────────────────────────────────────
    def _shap_mean(model, X_t):
        exp = shap.TreeExplainer(model)
        sv  = exp.shap_values(X_t)
        if isinstance(sv, list):
            sv = sv[1]
        elif sv.ndim == 3:
            sv = sv[:, :, 1]
        return np.abs(sv).mean(axis=0)

    shap_rf  = _shap_mean(rf,   X_test)
    shap_xgb = _shap_mean(xgb,  X_test)
    shap_lgb = _shap_mean(lgbm, X_test)

    # ── Mutual Information ────────────────────────────────────────────────────
    mi = mutual_info_classif(X_train, y_train, random_state=42)

    # ── Normalise ─────────────────────────────────────────────────────────────
    def _norm(arr):
        rng = arr.max() - arr.min()
        return (arr - arr.min()) / rng if rng > 0 else arr

    comp_df = pd.DataFrame({
        "Feature":           list(X.columns),
        "Gini (RF)":         _norm(rf.feature_importances_),
        "SHAP (RF)":         _norm(shap_rf),
        "SHAP (XGB)":        _norm(shap_xgb),
        "SHAP (LGB)":        _norm(shap_lgb),
        "Mutual_Information":_norm(mi),
    }).sort_values("SHAP (RF)", ascending=False)

    print(comp_df.to_string(index=False))

    # ── Plot ──────────────────────────────────────────────────────────────────
    n_feat = len(comp_df)
    x      = np.arange(n_feat)
    width  = 0.15

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - 2*width, comp_df["Gini (RF)"],          width, label="Gini (RF)",          color="#6366f1")
    ax.bar(x - 1*width, comp_df["SHAP (RF)"],          width, label="SHAP (RF)",          color="#f59e0b")
    ax.bar(x,            comp_df["SHAP (XGB)"],         width, label="SHAP (XGB)",         color="#10b981")
    ax.bar(x + 1*width, comp_df["SHAP (LGB)"],          width, label="SHAP (LightGBM)",    color="#ef4444")
    ax.bar(x + 2*width, comp_df["Mutual_Information"],  width, label="Mutual Information", color="#8b5cf6")

    ax.set_xticks(x)
    ax.set_xticklabels(comp_df["Feature"], rotation=40, ha="right")
    ax.set_ylabel("Normalised Score (0–1)")
    ax.set_title("Feature Importance: Gini vs SHAP (RF/XGB/LGB) vs Mutual Information")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("output_main/feature_importance_comparison.png", dpi=200, bbox_inches="tight")
    print("\n -> Saved output_main/feature_importance_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
