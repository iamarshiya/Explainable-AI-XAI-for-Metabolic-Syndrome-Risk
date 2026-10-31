"""
MetaRisk XAI — Clinician Portal (Upgraded)
============================================
New vs v1:
  • Extended fairness audit: gender + ethnicity + Equal Opportunity Difference
  • ROC curve panel for all 3 models
  • Global SHAP summary plot inline
  • LightGBM added to multi-model benchmark
  • Full SOAP report includes survival + conformal
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.metrics import RocCurveDisplay, roc_auc_score, accuracy_score
from google import genai
from google.genai import types

from utils import (
    load_assets_and_models,
    generate_pdf_report,
    apply_custom_css,
    compute_conformal_prediction_set,
    run_fairness_audit,
    get_ada_guideline_context,
    simulate_longitudinal_survival,
    GEMINI_API_KEY,
    EPS,
)

st.set_page_config(
    page_title="Clinician Portal — MetaRisk XAI",
    page_icon="🩺",
    layout="wide",
)
apply_custom_css()

# ─── Load assets ─────────────────────────────────────────────────────────────
(rf_model, xgb_model, lgb_model, best_model,
 df_data, X_train, X_test, y_test, feature_cols) = load_assets_and_models()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.header("📁 Cohort Configuration")
patient_idx = st.sidebar.number_input(
    "Select Patient Row Index",
    min_value=0, max_value=len(df_data) - 1, value=5
)

X_full = df_data[feature_cols]
selected_patient_row = X_full.iloc[patient_idx]
actual_outcome       = df_data.iloc[patient_idx]["Outcome"]

# ─── Header ──────────────────────────────────────────────────────────────────
st.title("🩺 Clinician Portal — Population Analytics & Fairness Audit")
st.markdown("""
<div class="disclaimer-banner">
⚠️ <b>Clinical Warning:</b> This decision support framework assists licensed healthcare providers
with algorithmic risk stratification and fairness audits. Final diagnosis remains solely the
clinical responsibility of the attending physician.
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ─── Patient quick-stats ─────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
def _box(label, value):
    return f'<div class="metric-box"><span style="color:#94a3b8;">{label}</span><h2 style="color:#f3f4f6;">{value}</h2></div>'

c1.markdown(_box("Age",  int(selected_patient_row["Age"])),           unsafe_allow_html=True)
c2.markdown(_box("BMI",  round(selected_patient_row["BMI"], 1)),       unsafe_allow_html=True)
c3.markdown(_box("BP",   int(selected_patient_row["BloodPressure"])), unsafe_allow_html=True)
c4.markdown(_box("Insulin", round(selected_patient_row["Insulin"], 1)), unsafe_allow_html=True)
s_color = "#f87171" if actual_outcome == 1 else "#34d399"
s_text  = "High Risk" if actual_outcome == 1 else "Low Risk"
c5.markdown(f'<div class="metric-box"><span style="color:#94a3b8;">Status</span><h2 style="color:{s_color};font-size:1.3rem;">{s_text}</h2></div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ─── Pre-compute ─────────────────────────────────────────────────────────────
pt_df   = pd.DataFrame([selected_patient_row])
rf_p    = rf_model.predict_proba(pt_df)[0][1]
xgb_p   = xgb_model.predict_proba(pt_df)[0][1]
lgb_p   = lgb_model.predict_proba(pt_df)[0][1]
best_p  = best_model.predict_proba(pt_df)[0][1]

conformal_info  = compute_conformal_prediction_set(best_model, X_train, pt_df)
survival_metrics= simulate_longitudinal_survival(
    selected_patient_row["Age"],
    selected_patient_row["BMI"],
    selected_patient_row["BloodPressure"],
)
ada_context = get_ada_guideline_context(
    selected_patient_row["BMI"],
    selected_patient_row["BloodPressure"],
    130.0,
)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
(tab_eval, tab_fairness, tab_roc,
 tab_shap_global, tab_multimodel, tab_genai) = st.tabs([
    "🎯 Diagnostic",
    "⚖️ Fairness Audit",
    "📈 ROC Curves",
    "🔍 SHAP Global",
    "🔬 Model Benchmark",
    "🤖 SOAP Notes",
])

# ── Diagnostic ────────────────────────────────────────────────────────────────
with tab_eval:
    st.markdown('<div class="luxury-card">', unsafe_allow_html=True)
    col_text, col_plot = st.columns(2)
    with col_text:
        st.subheader("Diagnostic & Conformal Evaluation")
        if best_p > 0.5:
            st.error(f"**Best Model — High Risk Detected**  Probability: **{best_p:.2%}**")
        else:
            st.success(f"**Best Model — Low Risk Status**  Probability: **{best_p:.2%}**")

        st.markdown(f"""
        <div style="background:rgba(31,41,55,.8);border:1px solid {conformal_info['color']};
                    padding:12px;border-radius:10px;margin-top:10px;">
            <span style="color:#94a3b8;font-size:.85rem;">CONFORMAL SET (95% COVERAGE)</span>
            <h4 style="color:{conformal_info['color']};margin:4px 0 0;">{conformal_info['set']}</h4>
            <p style="color:#d1d5db;font-size:.9rem;margin:4px 0 0;">{conformal_info['status']}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Survival Horizons (MetS-Free):**")
        st.write(f"• 1-Yr: {survival_metrics['1-Year Horizon']}  |  "
                 f"3-Yr: {survival_metrics['3-Year Horizon']}  |  "
                 f"5-Yr: {survival_metrics['5-Year Horizon']}")

    with col_plot:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor("#161b22")
        ax.set_facecolor("#0b0f19")
        sample  = df_data.sample(min(200, len(df_data)), random_state=42)
        sns.scatterplot(
            data=sample, x="BMI", y="BloodPressure", hue="Outcome",
            palette="coolwarm", alpha=0.65, ax=ax,
        )
        ax.scatter([selected_patient_row["BMI"]], [selected_patient_row["BloodPressure"]],
                   color="#fbbf24", s=200, marker="*", label="Selected", zorder=5)
        ax.tick_params(colors="#94a3b8")
        ax.xaxis.label.set_color("#94a3b8")
        ax.yaxis.label.set_color("#94a3b8")
        ax.legend(fontsize="x-small", facecolor="#161b22", labelcolor="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")
        st.pyplot(fig, clear_figure=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Fairness Audit (Extended) ─────────────────────────────────────────────────
with tab_fairness:
    st.markdown('<div class="luxury-card">', unsafe_allow_html=True)
    st.subheader("⚖️ Algorithmic Fairness — Gender & Ethnicity Audit")
    st.markdown("""
    Reporting **Selection Rate**, **True Positive Rate (Recall)**, and **False Positive Rate**
    across demographic sub-populations (gender + ethnicity from NHANES RIDRETH3 codes).
    Equal Opportunity Difference = max(TPR) − min(TPR).
    """)
    try:
        fairness_dict = run_fairness_audit(df_data, best_model, feature_cols)

        for attr, fair_df in fairness_dict.items():
            st.markdown(f"#### {attr.title()} Breakdown")
            st.dataframe(fair_df.style.format({
                "Selection Rate":     "{:.3f}",
                "True Positive Rate": "{:.3f}",
                "False Positive Rate":"{:.3f}",
            }), use_container_width=True)
            if "True Positive Rate" in fair_df.columns:
                tprs = fair_df["True Positive Rate"].dropna()
                eod  = tprs.max() - tprs.min()
                color = "green" if eod < 0.1 else ("orange" if eod < 0.2 else "red")
                st.markdown(f"**Equal Opportunity Difference:** :{color}[{eod:.3f}]"
                            f"  ({'Good ✅' if eod < 0.1 else 'Moderate ⚠️' if eod < 0.2 else 'High ❌'})")

            # Bar chart
            fig_f, ax_f = plt.subplots(figsize=(8, 3))
            fig_f.patch.set_facecolor("#161b22")
            ax_f.set_facecolor("#0b0f19")
            x = np.arange(len(fair_df))
            w = 0.25
            col_name = list(fair_df.columns)[0]
            ax_f.bar(x - w, fair_df.get("Selection Rate", 0),     width=w, label="Selection Rate",     color="#6366f1")
            ax_f.bar(x,     fair_df.get("True Positive Rate", 0), width=w, label="True Positive Rate", color="#10b981")
            ax_f.bar(x + w, fair_df.get("False Positive Rate",0), width=w, label="False Positive Rate",color="#ef4444")
            ax_f.set_xticks(x)
            ax_f.set_xticklabels(fair_df[col_name], rotation=30, ha="right", color="#94a3b8")
            ax_f.tick_params(colors="#94a3b8")
            ax_f.legend(facecolor="#161b22", labelcolor="white", fontsize="small")
            for sp in ax_f.spines.values():
                sp.set_edgecolor("#30363d")
            st.pyplot(fig_f, clear_figure=True)
            st.markdown("---")

    except Exception as e:
        st.warning(f"Fairness audit error: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# ── ROC Curves (NEW) ──────────────────────────────────────────────────────────
with tab_roc:
    st.markdown('<div class="luxury-card">', unsafe_allow_html=True)
    st.subheader("📈 ROC Curves — All Models on Holdout Test Set")
    fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
    fig_roc.patch.set_facecolor("#161b22")
    ax_roc.set_facecolor("#0b0f19")

    for name, mdl, color in [
        ("Random Forest", rf_model,  "#6366f1"),
        ("XGBoost",       xgb_model, "#f59e0b"),
        ("LightGBM",      lgb_model, "#10b981"),
    ]:
        try:
            RocCurveDisplay.from_predictions(
                y_test,
                mdl.predict_proba(X_test)[:, 1],
                name=f"{name} (AUC={roc_auc_score(y_test, mdl.predict_proba(X_test)[:,1]):.3f})",
                ax=ax_roc,
                color=color,
            )
        except Exception:
            pass

    ax_roc.plot([0, 1], [0, 1], "w--", alpha=0.4, label="Random")
    ax_roc.tick_params(colors="#94a3b8")
    ax_roc.xaxis.label.set_color("#94a3b8")
    ax_roc.yaxis.label.set_color("#94a3b8")
    ax_roc.legend(facecolor="#161b22", labelcolor="white")
    for sp in ax_roc.spines.values():
        sp.set_edgecolor("#30363d")
    st.pyplot(fig_roc, clear_figure=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Global SHAP (NEW) ─────────────────────────────────────────────────────────
with tab_shap_global:
    st.markdown('<div class="luxury-card">', unsafe_allow_html=True)
    st.subheader("🔍 Global SHAP Feature Importance")
    with st.spinner("Computing SHAP values for test cohort (may take a moment)..."):
        try:
            raw = best_model.calibrated_classifiers_[0].estimator
        except AttributeError:
            raw = best_model
        explainer = shap.TreeExplainer(raw)
        sample_n  = min(200, len(X_test))
        X_shap    = X_test.iloc[:sample_n]
        sv        = explainer.shap_values(X_shap)
        if isinstance(sv, list):
            sv_risk = sv[1]
        elif sv.ndim == 3:
            sv_risk = sv[:, :, 1]
        else:
            sv_risk = sv

    fig_s, ax_s = plt.subplots(figsize=(9, 5))
    fig_s.patch.set_facecolor("#161b22")
    shap.summary_plot(sv_risk, X_shap, show=False, plot_size=(9, 5))
    plt.title("SHAP Beeswarm — Population Feature Impact", color="#f3f4f6")
    plt.tight_layout()
    st.pyplot(fig_s, clear_figure=True)
    st.markdown("**Mean |SHAP| per feature:**")
    shap_means = pd.DataFrame({
        "Feature": list(X_shap.columns),
        "Mean |SHAP|": np.abs(sv_risk).mean(axis=0),
    }).sort_values("Mean |SHAP|", ascending=False)
    st.dataframe(shap_means, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Multi-Model Benchmark ─────────────────────────────────────────────────────
with tab_multimodel:
    st.markdown('<div class="luxury-card">', unsafe_allow_html=True)
    st.subheader("🔬 Side-by-Side Classifier Benchmark")
    rows = []
    for name, mdl in [("Random Forest", rf_model), ("XGBoost", xgb_model), ("LightGBM", lgb_model)]:
        try:
            acc = accuracy_score(y_test, mdl.predict(X_test))
            auc = roc_auc_score(y_test, mdl.predict_proba(X_test)[:, 1])
            p_pt= mdl.predict_proba(pt_df)[0][1]
            rows.append({
                "Model": name,
                "Holdout Accuracy": f"{acc:.4f}",
                "Holdout AUC":      f"{auc:.4f}",
                "Patient Risk":     f"{p_pt:.2%}",
            })
        except Exception:
            pass
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── GenAI SOAP Notes ──────────────────────────────────────────────────────────
with tab_genai:
    st.markdown('<div class="luxury-card">', unsafe_allow_html=True)
    st.subheader("🤖 ADA-Grounded SOAP Clinical Report (Gemini 2.5 Flash)")
    st.markdown(f"**ADA RAG Context:**\n> {ada_context}")

    api_key = GEMINI_API_KEY or st.secrets.get("GEMINI_API_KEY", "")

    if st.button("Generate SOAP Report", type="primary"):
        if not api_key:
            st.error("No GEMINI_API_KEY found. Add it to .env or Streamlit secrets.")
        else:
            with st.spinner("Querying Gemini with ADA RAG telemetry..."):
                try:
                    client  = genai.Client(api_key=api_key)
                    payload = {
                        "patient_index":     int(patient_idx),
                        "risk_probability":  round(float(best_p), 3),
                        "conformal_set":     conformal_info["set"],
                        "survival_horizons": survival_metrics,
                        "vitals":            selected_patient_row.to_dict(),
                    }
                    prompt = (
                        f"Official ADA Standards RAG:\n{ada_context}\n\n"
                        f"Patient telemetry:\n{payload}\n\n"
                        "Provide:\n"
                        "1. DOCTOR VIEW — SOAP Note (Subjective, Objective, Assessment, Plan) "
                        "with clinical justification, conformal bounds, and 5-year survival horizon.\n"
                        "2. PATIENT VIEW — Empathetic, plain-language actionable guidance "
                        "adhering to ADA lifestyle standards."
                    )
                    resp = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=(
                                "You are a rigorous clinical AI providing evidence-based summaries "
                                "grounded in ADA guidelines. Zero hallucinations on dosages."
                            ),
                            temperature=0.2,
                        ),
                    )
                    st.session_state.update({
                        "cl_narrative": resp.text,
                        "cl_risk":      best_p,
                        "cl_vitals":    selected_patient_row.to_dict(),
                        "cl_conformal": conformal_info,
                        "cl_survival":  survival_metrics,
                    })
                except Exception as e:
                    st.error(f"Gemini API error: {e}")

    if "cl_narrative" in st.session_state:
        st.markdown("---")
        st.markdown(st.session_state["cl_narrative"])
        pdf = generate_pdf_report(
            st.session_state["cl_vitals"], st.session_state["cl_risk"],
            st.session_state["cl_narrative"], st.session_state["cl_conformal"],
            st.session_state["cl_survival"],
        )
        st.download_button(
            "📥 Download SOAP PDF Report",
            data=pdf, file_name="MetaRisk_SOAP_Report.pdf", mime="application/pdf"
        )
    st.markdown('</div>', unsafe_allow_html=True)
