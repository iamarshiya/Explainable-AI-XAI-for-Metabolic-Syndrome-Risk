"""
MetaRisk XAI — Patient Portal (Upgraded)
=========================================
New vs v1:
  • Inline SHAP waterfall chart
  • Batch CSV upload → bulk predictions download
  • LightGBM in What-If comparison
  • DiCE counterfactuals (or heuristic fallback)
  • Fixed API key (reads from env / Streamlit secrets)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from google import genai

from utils import (
    load_assets_and_models,
    generate_pdf_report,
    apply_custom_css,
    compute_conformal_prediction_set,
    compute_counterfactual_recourse,
    get_ada_guideline_context,
    simulate_longitudinal_survival,
    render_shap_waterfall,
    GEMINI_API_KEY,
    EPS,
)

st.set_page_config(
    page_title="Patient Portal — MetaRisk XAI",
    page_icon="👤",
    layout="wide",
)
apply_custom_css()

# ─── Load assets ────────────────────────────────────────────────────────────
(rf_model, xgb_model, lgb_model, best_model,
 df_data, X_train, X_test, y_test, feature_cols) = load_assets_and_models()

# ─── Sidebar vitals ─────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Patient Vitals Entry")
age            = st.sidebar.slider("Age (Years)",               18.0, 100.0, 36.0,  1.0)
bmi            = st.sidebar.slider("Body Mass Index (BMI)",     10.0,  60.0, 31.4,  0.1)
blood_pressure = st.sidebar.slider("Blood Pressure (mm Hg)",    80.0, 200.0, 140.0, 1.0)
glucose        = st.sidebar.slider("Plasma Glucose (mg/dL)",    70.0, 300.0, 147.0, 1.0)
insulin        = st.sidebar.slider("Serum Insulin (μU/mL)",      2.0, 300.0, 208.5, 0.5)

waist_present = "WaistCircumference" in feature_cols
if waist_present:
    waist = st.sidebar.slider("Waist Circumference (cm)", 50.0, 160.0, 95.0, 0.5)

# Build input DataFrame matching feature_cols
def _build_input(age_v, bmi_v, bp_v, ins_v):
    row = {
        "Age":          age_v,
        "BMI":          bmi_v,
        "BloodPressure": bp_v,
        "Insulin":       ins_v,
        "BP_Age_ratio":  bp_v / (age_v + EPS),
        "BMI_Age_interaction":     bmi_v * age_v,
        "Insulin_BMI_interaction": ins_v * bmi_v,
        "Insulin_Glucose_ratio":   ins_v / (glucose + EPS),
    }
    if waist_present:
        row["WaistCircumference"] = waist if "waist" in dir() else 95.0
    return pd.DataFrame([[row[c] for c in feature_cols]], columns=feature_cols)

input_features = _build_input(age, bmi, blood_pressure, insulin)

# ─── Header ─────────────────────────────────────────────────────────────────
st.title("👤 Patient Portal — ADA-Grounded XAI & Risk Analytics")
st.markdown("""
<div class="disclaimer-banner">
⚠️ <b>Medical Disclaimer:</b> This research framework provides automated risk estimation
and ADA guideline-referenced insights. It does not replace professional medical advice.
Always consult a licensed physician.
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ─── Pre-compute ─────────────────────────────────────────────────────────────
risk_proba      = best_model.predict_proba(input_features)[0][1]
conformal_info  = compute_conformal_prediction_set(best_model, X_train, input_features)
survival_metrics= simulate_longitudinal_survival(age, bmi, blood_pressure)
ada_context     = get_ada_guideline_context(bmi, blood_pressure, glucose)

# ─── Tabs ────────────────────────────────────────────────────────────────────
(tab_diag, tab_waterfall, tab_survival, tab_sandbox,
 tab_recourse, tab_batch, tab_report) = st.tabs([
    "📊 Diagnostic",
    "🔍 SHAP Waterfall",
    "📈 Survival",
    "🧪 What-If",
    "⚡ Recourse",
    "📂 Batch Upload",
    "🤖 ADA Report",
])

# ── Tab 1: Diagnostic ────────────────────────────────────────────────────────
with tab_diag:
    col_risk, col_summary = st.columns(2, gap="medium")
    with col_risk:
        st.markdown("### Risk Assessment & Conformal Bounds")
        badge_class = "high-risk-badge" if risk_proba > 0.5 else "low-risk-badge"
        label_color = "#f87171" if risk_proba > 0.5 else "#34d399"
        label_text  = "Elevated Health Alert" if risk_proba > 0.5 else "Optimal Health Range"
        st.markdown(f"""
        <div class="{badge_class}">
            <span style="color:{label_color};font-weight:600;">{label_text}</span>
            <h1 style="color:#fff;font-size:3rem;margin:8px 0;">{risk_proba:.1%}</h1>
            <p style="color:{label_color};margin:0;">MetS Risk Probability (Best Calibrated Model)</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="luxury-card" style="margin-top:15px;padding:15px;">
            <h4 style="margin:0 0 5px;color:#94a3b8;font-size:.95rem;">
                🔒 Split Conformal Guarantee (95% Coverage)</h4>
            <p style="margin:0;font-size:1.05rem;color:{conformal_info['color']};">
                <b>Prediction Set:</b> {conformal_info['set']}</p>
            <p style="margin:5px 0 0;font-size:.85rem;color:#94a3b8;">{conformal_info['status']}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_summary:
        st.markdown(f"""
        <div class="luxury-card" style="height:100%;">
            <h3 style="margin-top:0;color:#94a3b8;">Patient Summary</h3>
            <hr style="border-color:rgba(255,255,255,.08);">
            <ul style="list-style:none;padding:0;line-height:2.4rem;">
                <li>🎯 <b>Age:</b> {age:.0f} yrs</li>
                <li>⚖️ <b>BMI:</b> {bmi:.1f} kg/m²</li>
                <li>🩸 <b>Glucose:</b> {glucose:.0f} mg/dL</li>
                <li>💓 <b>Blood Pressure:</b> {blood_pressure:.0f} mm Hg</li>
                <li>🧪 <b>Insulin:</b> {insulin:.1f} μU/mL</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Model comparison bar
    st.markdown("---")
    st.subheader("All-Model Risk Comparison")
    rf_p  = rf_model.predict_proba(input_features)[0][1]
    xgb_p = xgb_model.predict_proba(input_features)[0][1]
    lgb_p = lgb_model.predict_proba(input_features)[0][1]
    cmp_df = pd.DataFrame({
        "Model": ["Random Forest", "XGBoost", "LightGBM"],
        "Risk %": [rf_p * 100, xgb_p * 100, lgb_p * 100],
    })
    fig, ax = plt.subplots(figsize=(7, 2.5))
    fig.patch.set_facecolor("#161b22")
    ax.set_facecolor("#0b0f19")
    colors = ["#ef4444" if v > 50 else "#10b981" for v in cmp_df["Risk %"]]
    ax.barh(cmp_df["Model"], cmp_df["Risk %"], color=colors)
    ax.axvline(50, color="#fbbf24", linestyle="--", linewidth=1)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Risk Probability (%)", color="#94a3b8")
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    st.pyplot(fig, clear_figure=True)

# ── Tab 2: SHAP Waterfall (NEW) ───────────────────────────────────────────────
with tab_waterfall:
    st.markdown('<div class="luxury-card">', unsafe_allow_html=True)
    st.subheader("🔍 SHAP Waterfall — Feature-Level Explanation")
    st.markdown("Breaks down each feature's individual push toward or away from High Risk for *this specific patient*.")
    with st.spinner("Computing SHAP values..."):
        wf_fig = render_shap_waterfall(best_model, X_train, input_features)
    st.pyplot(wf_fig, clear_figure=True)
    st.markdown("**How to read:** Red bars push risk UP, blue bars push risk DOWN. The base value is the average model output across all patients.")
    st.markdown('</div>', unsafe_allow_html=True)

# ── Tab 3: Survival ───────────────────────────────────────────────────────────
with tab_survival:
    st.markdown('<div class="luxury-card">', unsafe_allow_html=True)
    st.subheader("📈 Longitudinal MetS-Free Survival Horizons")
    c1, c2, c3 = st.columns(3)
    c1.metric("1-Year", survival_metrics["1-Year Horizon"])
    c2.metric("3-Year", survival_metrics["3-Year Horizon"])
    c3.metric("5-Year", survival_metrics["5-Year Horizon"])

    # Visual bar
    years  = [1, 3, 5]
    probs  = [float(survival_metrics[f"{y}-Year Horizon"].strip("%")) / 100 for y in years]
    fig2, ax2 = plt.subplots(figsize=(6, 2.5))
    fig2.patch.set_facecolor("#161b22")
    ax2.set_facecolor("#0b0f19")
    ax2.bar(["1 Year", "3 Years", "5 Years"], probs, color=["#34d399", "#f59e0b", "#ef4444"])
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("MetS-Free Probability", color="#94a3b8")
    ax2.tick_params(colors="#94a3b8")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#30363d")
    st.pyplot(fig2, clear_figure=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Tab 4: What-If Sandbox ────────────────────────────────────────────────────
with tab_sandbox:
    st.markdown('<div class="luxury-card">', unsafe_allow_html=True)
    st.subheader("🧪 What-If Biomarker Scenario Sandbox")
    sb1, sb2 = st.columns(2)
    sim_bmi = sb1.slider("Simulated BMI",           10.0, 60.0, float(bmi), 0.1,  key="s_bmi")
    sim_bp  = sb1.slider("Simulated Blood Pressure", 80.0,200.0, float(blood_pressure),1.0,key="s_bp")
    sim_ins = sb2.slider("Simulated Insulin",          2.0,300.0, float(insulin), 0.5,key="s_ins")
    sim_age = sb2.slider("Simulated Age",             18.0,100.0, float(age),     1.0, key="s_age")

    sim_feat = _build_input(sim_age, sim_bmi, sim_bp, sim_ins)
    sim_rf   = rf_model.predict_proba(sim_feat)[0][1]
    sim_xgb  = xgb_model.predict_proba(sim_feat)[0][1]
    sim_lgb  = lgb_model.predict_proba(sim_feat)[0][1]
    sim_best = best_model.predict_proba(sim_feat)[0][1]
    delta    = sim_best - risk_proba

    st.markdown("---")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Baseline (Best Model)", f"{risk_proba:.1%}")
    r2.metric("Simulated (Best Model)", f"{sim_best:.1%}", delta=f"{delta:.1%}", delta_color="inverse")
    r3.metric("RF Simulated",   f"{sim_rf:.1%}")
    r4.metric("LightGBM Simul.", f"{sim_lgb:.1%}")
    st.markdown('</div>', unsafe_allow_html=True)

# ── Tab 5: Counterfactual Recourse ────────────────────────────────────────────
with tab_recourse:
    st.markdown('<div class="luxury-card">', unsafe_allow_html=True)
    st.subheader("⚡ Counterfactual Recourse Pathway (DiCE / Heuristic)")
    vitals_dict = {
        "Age": age, "BMI": bmi, "BloodPressure": blood_pressure, "Insulin": insulin,
        "BP_Age_ratio": blood_pressure / (age + EPS),
        "BMI_Age_interaction": bmi * age,
        "Insulin_BMI_interaction": insulin * bmi,
        "Insulin_Glucose_ratio": insulin / (glucose + EPS),
    }
    if waist_present:
        vitals_dict["WaistCircumference"] = waist if "waist" in dir() else 95.0
    recourse = compute_counterfactual_recourse(best_model, vitals_dict, feature_cols)
    st.markdown(f"""
    <div style="background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.3);
                padding:16px;border-radius:12px;color:#a7f3d0;font-size:1.05rem;">
        {recourse}
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Tab 6: Batch Upload (NEW) ─────────────────────────────────────────────────
with tab_batch:
    st.markdown('<div class="luxury-card">', unsafe_allow_html=True)
    st.subheader("📂 Batch Patient Risk Assessment")
    st.markdown("""
    Upload a CSV file with columns: `Age, BMI, BloodPressure, Insulin, Glucose`
    (and optionally `WaistCircumference`). The system will compute MetS risk for each row.
    """)
    uploaded = st.file_uploader("Upload patient CSV", type=["csv"])
    if uploaded:
        try:
            batch_df = pd.read_csv(uploaded)
            st.success(f"Loaded {len(batch_df)} patient records.")
            st.dataframe(batch_df.head(5), use_container_width=True)

            if st.button("Run Batch Prediction", type="primary"):
                with st.spinner("Computing risk scores..."):
                    def _row_to_features(row):
                        age_v = row.get("Age", 40)
                        bmi_v = row.get("BMI", 25)
                        bp_v  = row.get("BloodPressure", 120)
                        ins_v = row.get("Insulin", 15)
                        glu_v = row.get("Glucose", 100)
                        feat  = {
                            "Age":          age_v,
                            "BMI":          bmi_v,
                            "BloodPressure": bp_v,
                            "Insulin":       ins_v,
                            "BP_Age_ratio":  bp_v / (age_v + EPS),
                            "BMI_Age_interaction":     bmi_v * age_v,
                            "Insulin_BMI_interaction": ins_v * bmi_v,
                            "Insulin_Glucose_ratio":   ins_v / (glu_v + EPS),
                        }
                        if waist_present:
                            feat["WaistCircumference"] = row.get("WaistCircumference", 95.0)
                        return {c: feat[c] for c in feature_cols if c in feat}

                    records = [_row_to_features(r) for _, r in batch_df.iterrows()]
                    feat_df = pd.DataFrame(records, columns=[c for c in feature_cols if c in records[0]])
                    probs   = best_model.predict_proba(feat_df)[:, 1]
                    preds   = ["High Risk" if p > 0.5 else "Low Risk" for p in probs]

                    out_df = batch_df.copy()
                    out_df["Risk_Probability"] = [f"{p:.2%}" for p in probs]
                    out_df["Risk_Label"]       = preds

                st.success("Batch predictions complete!")
                st.dataframe(out_df, use_container_width=True)
                csv_bytes = out_df.to_csv(index=False).encode()
                st.download_button(
                    "📥 Download Results CSV",
                    data=csv_bytes,
                    file_name="metarisk_batch_results.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Error processing file: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# ── Tab 7: ADA Report ─────────────────────────────────────────────────────────
with tab_report:
    st.markdown('<div class="luxury-card">', unsafe_allow_html=True)
    st.subheader("🤖 ADA-Grounded Generative Clinical Narrative")
    st.markdown(f"**Retrieved ADA Guidelines Context:**\n> {ada_context}")

    api_key = GEMINI_API_KEY or st.secrets.get("GEMINI_API_KEY", "")

    if st.button("Generate ADA Narrative + PDF", type="primary"):
        if not api_key:
            st.error("No GEMINI_API_KEY found. Add it to .env or Streamlit secrets.")
        else:
            with st.spinner("Querying Gemini 2.5 Flash with ADA RAG context..."):
                try:
                    client = genai.Client(api_key=api_key)
                    prompt = (
                        f"Using these ADA Standards of Care:\n{ada_context}\n\n"
                        f"Patient vitals: Age={age}, BMI={bmi}, BP={blood_pressure}, "
                        f"Glucose={glucose}, Insulin={insulin}. "
                        f"MetS risk probability: {risk_proba:.2%}.\n\n"
                        "Provide:\n"
                        "1. DOCTOR VIEW: Structured SOAP Note (Subjective, Objective, Assessment, Plan) "
                        "with clinical justification, conformal bounds, and survival horizons.\n"
                        "2. PATIENT VIEW: Empathetic, plain-language actionable guidance.\n"
                        "Ensure zero hallucinations regarding dosages or contraindications."
                    )
                    resp = client.models.generate_content(
                        model="gemini-2.5-flash", contents=prompt
                    )
                    narrative = resp.text
                    st.session_state.update({
                        "p_narrative": narrative,
                        "p_risk":      risk_proba,
                        "p_vitals":    {"Age": age, "BMI": bmi, "BP": blood_pressure,
                                        "Glucose": glucose, "Insulin": insulin},
                        "p_conformal": conformal_info,
                        "p_survival":  survival_metrics,
                    })
                except Exception as e:
                    st.error(f"Gemini API error: {e}")

    if "p_narrative" in st.session_state:
        st.markdown("---")
        st.markdown(st.session_state["p_narrative"])
        pdf = generate_pdf_report(
            st.session_state["p_vitals"], st.session_state["p_risk"],
            st.session_state["p_narrative"], st.session_state["p_conformal"],
            st.session_state["p_survival"],
        )
        st.download_button(
            "📥 Download Clinical PDF Report",
            data=pdf, file_name="MetaRisk_Patient_Report.pdf", mime="application/pdf"
        )
    st.markdown('</div>', unsafe_allow_html=True)
