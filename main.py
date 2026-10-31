"""
MetaRisk XAI — Upgraded Training Pipeline
==========================================
Upgrades over v1:
  • Proper ATP-III-proxy MetS label (glucose + BMI + BP composite criteria)
  • LightGBM added alongside Random Forest and XGBoost
  • Optuna hyperparameter tuning for all 3 models
  • Probability calibration (CalibratedClassifierCV)
  • Extended SHAP output: summary + waterfall + dependence plots
  • LIME explanation saved
  • Best calibrated model serialised as model.pkl
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular
import optuna
import lightgbm as lgb

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score,
    RocCurveDisplay, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

os.makedirs("output_main", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def fetch_nhanes_data():
    """Merge 5 NHANES-J SAS modules into one clinical DataFrame."""
    print("1. Loading NHANES Multi-Year Population Data from local 'data/' directory...")
    data_dir = "data"

    def path(name):
        for ext in (".xpt", ".XPT"):
            p = os.path.join(data_dir, name + ext)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(
            f"Cannot find {name} in data/. "
            "Download NHANES-J files (DEMO_J, BMX_J, BPX_J, GLU_J, INS_J) "
            "from https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017"
        )

    df_demo  = pd.read_sas(path("DEMO_J"))[["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3"]]
    df_bmx   = pd.read_sas(path("BMX_J"))[["SEQN", "BMXBMI", "BMXWAIST"]]
    df_bpx   = pd.read_sas(path("BPX_J"))[["SEQN", "BPXSY1"]]
    df_glu   = pd.read_sas(path("GLU_J"))[["SEQN", "LBXGLU"]]
    df_ins   = pd.read_sas(path("INS_J"))[["SEQN", "LBXIN"]]

    df = (df_demo
          .merge(df_bmx, on="SEQN", how="inner")
          .merge(df_bpx, on="SEQN", how="inner")
          .merge(df_glu, on="SEQN", how="inner")
          .merge(df_ins, on="SEQN", how="left"))

    df.rename(columns={
        "RIDAGEYR": "Age",
        "RIAGENDR": "Gender",
        "RIDRETH3": "Ethnicity",
        "BMXBMI":   "BMI",
        "BMXWAIST": "WaistCircumference",
        "BPXSY1":   "BloodPressure",
        "LBXGLU":   "Glucose",
        "LBXIN":    "Insulin",
    }, inplace=True)
    df.drop(columns=["SEQN"], inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PREPROCESSING + ATP-III PROXY LABEL
# ─────────────────────────────────────────────────────────────────────────────

def prepare_clinical_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    eps = 1e-6

    # --- impute ---
    for col in ["Glucose", "BloodPressure", "Insulin", "BMI", "WaistCircumference", "Age"]:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
            # group-median then global-median fallback
            df[col] = df[col].fillna(df[col].median())

    # --- ATP-III proxy MetS label (3-of-5 criteria) ---
    # Criterion 1: Abdominal obesity (waist ≥102 cm men / ≥88 cm women)
    # We use BMI≥30 as proxy when waist is missing
    gender_map = {1.0: "Male", 2.0: "Female"}
    df["_gender"] = df["Gender"].map(gender_map).fillna("Male")

    if "WaistCircumference" in df.columns:
        waist_thresh = np.where(df["_gender"] == "Male", 102.0, 88.0)
        df["crit_waist"] = (df["WaistCircumference"] >= waist_thresh).astype(int)
    else:
        df["crit_waist"] = (df["BMI"] >= 30.0).astype(int)

    # Criterion 2: High BP (≥130 systolic or already on medication proxy)
    df["crit_bp"]      = (df["BloodPressure"] >= 130.0).astype(int)
    # Criterion 3: Elevated fasting glucose ≥100 mg/dL
    df["crit_glucose"] = (df["Glucose"] >= 100.0).astype(int)
    # Criterion 4: High BMI as proxy for low HDL / high triglycerides
    df["crit_lipid_proxy"] = (df["BMI"] >= 27.5).astype(int)
    # Criterion 5: Insulin resistance proxy (high insulin)
    df["crit_insulin"] = (df["Insulin"] >= 25.0).astype(int)

    criteria_sum = (df["crit_waist"] + df["crit_bp"] + df["crit_glucose"] +
                    df["crit_lipid_proxy"] + df["crit_insulin"])
    df["Outcome"] = (criteria_sum >= 3).astype(int)  # ATP-III: 3-of-5

    # Also keep a strict diabetic flag as auxiliary
    df["Diabetic"] = (df["Glucose"] >= 126.0).astype(int)

    # --- engineered interaction features ---
    df["Glucose_BMI_interaction"]  = df["Glucose"] * df["BMI"]
    df["Insulin_Glucose_ratio"]    = df["Insulin"] / (df["Glucose"] + eps)
    df["BP_Age_ratio"]             = df["BloodPressure"] / (df["Age"] + eps)
    df["BMI_Age_interaction"]      = df["BMI"] * df["Age"]
    df["Insulin_BMI_interaction"]  = df["Insulin"] * df["BMI"]

    # drop helper columns
    df.drop(columns=[c for c in ["crit_waist","crit_bp","crit_glucose",
                                  "crit_lipid_proxy","crit_insulin","_gender"]
                     if c in df.columns], inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FEATURE SELECTION
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "Age", "BMI", "BloodPressure", "Insulin",
    "BP_Age_ratio", "BMI_Age_interaction", "Insulin_BMI_interaction",
    "Insulin_Glucose_ratio",
]
# WaistCircumference added dynamically if present
DROP_COLS = ["Outcome", "Diabetic", "Glucose", "Glucose_BMI_interaction",
             "Ethnicity", "Gender", "SEQN"]


def build_feature_matrix(df: pd.DataFrame):
    cols = FEATURE_COLS.copy()
    if "WaistCircumference" in df.columns:
        cols.insert(2, "WaistCircumference")
    X = df[cols]
    y = df["Outcome"]
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 4.  OPTUNA TUNING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def tune_rf(X_tr, y_tr, n_trials=30):
    def objective(trial):
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 100, 600, step=100),
            "max_depth":       trial.suggest_int("max_depth", 3, 12),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
            "random_state": 42,
        }
        model = RandomForestClassifier(**params)
        return cross_val_score(model, X_tr, y_tr, cv=5, scoring="roc_auc",
                               n_jobs=-1).mean()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_xgb(X_tr, y_tr, n_trials=30):
    scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    def objective(trial):
        params = {
            "n_estimators":  trial.suggest_int("n_estimators", 100, 600, step=100),
            "max_depth":     trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":     trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "scale_pos_weight": scale_pos,
            "eval_metric": "logloss",
            "random_state": 42,
        }
        model = XGBClassifier(**params)
        return cross_val_score(model, X_tr, y_tr, cv=5, scoring="roc_auc",
                               n_jobs=-1).mean()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_lgb(X_tr, y_tr, n_trials=30):
    def objective(trial):
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 100, 600, step=100),
            "max_depth":       trial.suggest_int("max_depth", 3, 12),
            "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves":      trial.suggest_int("num_leaves", 20, 200),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample":       trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "verbose": -1,
        }
        model = lgb.LGBMClassifier(**params)
        return cross_val_score(model, X_tr, y_tr, cv=5, scoring="roc_auc",
                               n_jobs=-1).mean()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Load & prepare ──────────────────────────────────────────────────────
    df_raw   = fetch_nhanes_data()
    df_clean = prepare_clinical_data(df_raw)
    X, y     = build_feature_matrix(df_clean)

    print(f"\n✅ Dataset ready: {X.shape[0]} patients × {X.shape[1]} features")
    print(f"   MetS prevalence (ATP-III proxy): {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ── Optuna tuning ────────────────────────────────────────────────────────
    print("\n2. Running Optuna hyperparameter search (30 trials × 3 models)...")
    rf_params  = tune_rf(X_train,  y_train, n_trials=30)
    xgb_params = tune_xgb(X_train, y_train, n_trials=30)
    lgb_params = tune_lgb(X_train, y_train, n_trials=30)
    print("   Tuning complete.")

    # ── Train base models ────────────────────────────────────────────────────
    print("\n3. Training + calibrating models...")
    rf_base  = RandomForestClassifier(**rf_params, random_state=42)
    xgb_base = XGBClassifier(**xgb_params, eval_metric="logloss", random_state=42)
    lgb_base = lgb.LGBMClassifier(**lgb_params, random_state=42, verbose=-1)

    # Isotonic calibration via 5-fold CV (produces reliable probabilities)
    rf_cal  = CalibratedClassifierCV(rf_base,  method="isotonic", cv=5)
    xgb_cal = CalibratedClassifierCV(xgb_base, method="isotonic", cv=5)
    lgb_cal = CalibratedClassifierCV(lgb_base, method="isotonic", cv=5)

    rf_cal.fit(X_train,  y_train)
    xgb_cal.fit(X_train, y_train)
    lgb_cal.fit(X_train, y_train)

    # ── Evaluation ───────────────────────────────────────────────────────────
    print("\n4. Model Evaluation:")
    results = {}
    for name, model in [("Random Forest", rf_cal), ("XGBoost", xgb_cal), ("LightGBM", lgb_cal)]:
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        acc  = accuracy_score(y_test, y_pred)
        auc  = roc_auc_score(y_test, y_proba)
        results[name] = {"Accuracy": acc, "ROC-AUC": auc}
        print(f"   {name:20s} | Accuracy: {acc:.4f} | AUC: {auc:.4f}")

    best_model_name = max(results, key=lambda k: results[k]["ROC-AUC"])
    best_model = {"Random Forest": rf_cal, "XGBoost": xgb_cal, "LightGBM": lgb_cal}[best_model_name]
    print(f"\n   ✅ Best model: {best_model_name} (AUC={results[best_model_name]['ROC-AUC']:.4f})")

    joblib.dump(best_model, "model.pkl")
    joblib.dump(rf_cal,     "model_rf.pkl")
    joblib.dump(xgb_cal,    "model_xgb.pkl")
    joblib.dump(lgb_cal,    "model_lgb.pkl")
    print("   Saved model.pkl, model_rf.pkl, model_xgb.pkl, model_lgb.pkl")

    # ── ROC curve comparison ─────────────────────────────────────────────────
    print("\n5. Saving ROC comparison plot...")
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in [("RF", rf_cal), ("XGBoost", xgb_cal), ("LightGBM", lgb_cal)]:
        RocCurveDisplay.from_predictions(
            y_test, model.predict_proba(X_test)[:, 1], name=name, ax=ax
        )
    ax.set_title("ROC Curves — Model Comparison")
    plt.tight_layout()
    plt.savefig("output_main/roc_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ── Calibration curve ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, model in [("RF (cal)", rf_cal), ("XGB (cal)", xgb_cal), ("LGB (cal)", lgb_cal)]:
        prob_true, prob_pred = calibration_curve(
            y_test, model.predict_proba(X_test)[:, 1], n_bins=10
        )
        ax.plot(prob_pred, prob_true, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curves")
    ax.legend()
    plt.tight_layout()
    plt.savefig("output_main/calibration_curves.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ── SHAP on best model ───────────────────────────────────────────────────
    print("\n6. Generating SHAP explanations (best model)...")
    # SHAP works on the underlying uncalibrated base; extract it
    raw_model = best_model.calibrated_classifiers_[0].estimator
    explainer  = shap.TreeExplainer(raw_model)
    shap_vals  = explainer.shap_values(X_test)

    if isinstance(shap_vals, list):
        sv = shap_vals[1]
    elif shap_vals.ndim == 3:
        sv = shap_vals[:, :, 1]
    else:
        sv = shap_vals

    # Summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv, X_test, show=False)
    plt.title("SHAP Summary Plot — Global Feature Importance")
    plt.tight_layout()
    plt.savefig("output_main/shap_summary.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Waterfall for patient index 5
    patient_idx = 5
    patient_sv  = shap.Explanation(
        values       = sv[patient_idx],
        base_values  = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
        data         = X_test.iloc[patient_idx].values,
        feature_names= list(X_test.columns)
    )
    shap.plots.waterfall(patient_sv, show=False)
    plt.title(f"SHAP Waterfall — Patient #{patient_idx}")
    plt.tight_layout()
    plt.savefig("output_main/shap_waterfall_patient5.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Dependence plots for top-2 features
    mean_abs = np.abs(sv).mean(axis=0)
    top2 = np.argsort(mean_abs)[::-1][:2]
    for feat_idx in top2:
        fname = X_test.columns[feat_idx]
        shap.dependence_plot(feat_idx, sv, X_test, show=False)
        plt.title(f"SHAP Dependence — {fname}")
        plt.tight_layout()
        plt.savefig(f"output_main/shap_dependence_{fname}.png", dpi=200, bbox_inches="tight")
        plt.close()

    # ── LIME ─────────────────────────────────────────────────────────────────
    print("\n7. Generating LIME explanation for patient #5...")
    lime_exp = lime.lime_tabular.LimeTabularExplainer(
        training_data = X_train.values,
        feature_names = list(X_train.columns),
        class_names   = ["Low Risk", "High Risk"],
        mode          = "classification"
    )

    def predict_fn(x):
        return best_model.predict_proba(pd.DataFrame(x, columns=X_train.columns))

    exp = lime_exp.explain_instance(
        data_row   = X_test.iloc[patient_idx].values,
        predict_fn = predict_fn
    )
    exp.save_to_file("output_main/patient_lime_explanation.html")

    # ── Confusion matrix ─────────────────────────────────────────────────────
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(
        y_test, best_model.predict(X_test), ax=ax,
        display_labels=["Low Risk", "High Risk"], cmap="Blues"
    )
    ax.set_title(f"Confusion Matrix — {best_model_name}")
    plt.tight_layout()
    plt.savefig("output_main/confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("\n✅ Pipeline complete — all outputs saved to output_main/")
    print("   Files: model.pkl, model_rf.pkl, model_xgb.pkl, model_lgb.pkl")


if __name__ == "__main__":
    main()
