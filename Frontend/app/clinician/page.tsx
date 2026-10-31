"use client";
import { useEffect, useState } from "react";
import { BarChart2, Shield, Info, Loader2 } from "lucide-react";
import { getFairness, getBenchmark, type FairnessResult, type BenchmarkRow } from "@/lib/api";
import BenchmarkTable from "@/components/BenchmarkTable";
import FairnessChart  from "@/components/FairnessChart";

type TabId = "benchmark" | "fairness" | "about";

function Tab({
  active, onClick, icon: Icon, label,
}: { active: boolean; onClick: () => void; icon: React.ElementType; label: string }) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium rounded-lg transition-all
        ${active
          ? "bg-teal-600/20 text-teal-300 border border-teal-500/30"
          : "text-slate-500 hover:text-slate-300 hover:bg-slate-800/60"}`}
    >
      <Icon size={14} />
      {label}
    </button>
  );
}

export default function ClinicianDashboard() {
  const [tab, setTab] = useState<TabId>("benchmark");

  const [bench, setBench]     = useState<BenchmarkRow[] | null>(null);
  const [fair,  setFair]      = useState<FairnessResult | null>(null);
  const [loadB, setLoadB]     = useState(false);
  const [loadF, setLoadF]     = useState(false);
  const [errB,  setErrB]      = useState("");
  const [errF,  setErrF]      = useState("");

  useEffect(() => {
    if (tab === "benchmark" && !bench) {
      setLoadB(true);
      getBenchmark()
        .then(setBench)
        .catch(() => setErrB("API unavailable — showing demo data"))
        .finally(() => setLoadB(false));
    }
    if (tab === "fairness" && !fair) {
      setLoadF(true);
      getFairness()
        .then(setFair)
        .catch(() => setErrF("API unavailable — showing demo data"))
        .finally(() => setLoadF(false));
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab]);

  const DEMO_BENCH: BenchmarkRow[] = [
    { Model: "Logistic Regression", Accuracy: "0.7201 ± 0.0180", "ROC-AUC": "0.7823 ± 0.0210", "F1-Score": "0.6941 ± 0.0230" },
    { Model: "Random Forest",       Accuracy: "0.8301 ± 0.0120", "ROC-AUC": "0.8891 ± 0.0130", "F1-Score": "0.8102 ± 0.0140" },
    { Model: "Gradient Boosting",   Accuracy: "0.8201 ± 0.0130", "ROC-AUC": "0.8801 ± 0.0140", "F1-Score": "0.8001 ± 0.0150" },
    { Model: "XGBoost",             Accuracy: "0.8401 ± 0.0110", "ROC-AUC": "0.8991 ± 0.0120", "F1-Score": "0.8201 ± 0.0130" },
    { Model: "LightGBM",            Accuracy: "0.8501 ± 0.0100", "ROC-AUC": "0.9101 ± 0.0110", "F1-Score": "0.8301 ± 0.0120" },
  ];

  const DEMO_FAIR: FairnessResult = {
    gender: [
      { group: "Male",   selection_rate: 0.48, tpr: 0.82, fpr: 0.18, eod:  0.00 },
      { group: "Female", selection_rate: 0.41, tpr: 0.74, fpr: 0.21, eod: -0.08 },
    ],
    ethnicity: [
      { group: "Non-Hispanic White", selection_rate: 0.40, tpr: 0.79, fpr: 0.17 },
      { group: "Non-Hispanic Black", selection_rate: 0.52, tpr: 0.84, fpr: 0.23 },
      { group: "Hispanic",           selection_rate: 0.55, tpr: 0.86, fpr: 0.24 },
      { group: "Non-Hispanic Asian", selection_rate: 0.35, tpr: 0.72, fpr: 0.15 },
      { group: "Other/Multi",        selection_rate: 0.44, tpr: 0.78, fpr: 0.19 },
    ],
  };

  const benchData = bench ?? (errB ? DEMO_BENCH : null);
  const fairData  = fair  ?? (errF ? DEMO_FAIR  : null);

  return (
    <div className="mx-auto max-w-6xl px-6 py-10">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">Clinician Dashboard</h1>
        <p className="text-slate-400">
          Model performance benchmarks, demographic fairness audit, and system overview.
        </p>
      </div>

      {/* KPI strip */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        {[
          { label: "Best ROC-AUC",   value: "0.91",  sub: "LightGBM (calibrated)" },
          { label: "F1 Score",       value: "0.83",  sub: "10-fold stratified CV"  },
          { label: "EOD (Gender)",   value: "−0.08", sub: "Female vs Male TPR"     },
          { label: "Cohort",         value: "8 k+",  sub: "NHANES-J participants"  },
        ].map((k) => (
          <div key={k.label} className="glass p-4">
            <div className="text-2xl font-bold text-white mb-0.5">{k.value}</div>
            <div className="text-xs font-semibold text-teal-300">{k.label}</div>
            <div className="text-xs text-slate-500 mt-0.5">{k.sub}</div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <div className="flex gap-2 mb-4 flex-wrap">
        <Tab active={tab === "benchmark"} onClick={() => setTab("benchmark")} icon={BarChart2} label="Model Benchmark" />
        <Tab active={tab === "fairness"}  onClick={() => setTab("fairness")}  icon={Shield}    label="Fairness Audit"  />
        <Tab active={tab === "about"}     onClick={() => setTab("about")}     icon={Info}      label="About"          />
      </div>

      <div className="glass p-6 min-h-[420px]">

        {/* ─ Benchmark ─ */}
        {tab === "benchmark" && (
          <div>
            <h2 className="text-lg font-semibold text-white mb-1">10-Fold CV Model Benchmark</h2>
            <p className="text-xs text-slate-500 mb-6">Mean ± 2×Std across 10 stratified folds on NHANES-J cohort.</p>
            {loadB && <Loader2 size={24} className="animate-spin text-slate-500 mx-auto mt-16" />}
            {errB  && <p className="text-xs text-amber-400 mb-4">⚠ {errB}</p>}
            {benchData && <BenchmarkTable rows={benchData} />}
          </div>
        )}

        {/* ─ Fairness ─ */}
        {tab === "fairness" && (
          <div>
            <h2 className="text-lg font-semibold text-white mb-1">Demographic Fairness Audit</h2>
            <p className="text-xs text-slate-500 mb-6">
              Selection rate, TPR, FPR across gender and NHANES ethnicity groups.
              EOD = True Positive Rate difference vs. reference group.
            </p>
            {loadF && <Loader2 size={24} className="animate-spin text-slate-500 mx-auto mt-16" />}
            {errF  && <p className="text-xs text-amber-400 mb-4">⚠ {errF}</p>}
            {fairData && <FairnessChart data={fairData} />}
          </div>
        )}

        {/* ─ About ─ */}
        {tab === "about" && (
          <div className="space-y-6 text-sm text-slate-400 leading-relaxed max-w-3xl">
            <h2 className="text-lg font-semibold text-white">About MetaRisk XAI</h2>

            <div className="space-y-3">
              <h3 className="text-white font-medium">Label</h3>
              <p>
                Metabolic Syndrome defined using an <strong className="text-slate-200">ATP-III proxy</strong>: ≥3 of 4
                criteria — BMI ≥ 27.5 (abdominal obesity proxy), systolic BP ≥ 130 mmHg,
                fasting glucose ≥ 100 mg/dL, insulin ≥ 25 µU/mL.
              </p>
            </div>

            <div className="space-y-3">
              <h3 className="text-white font-medium">Models</h3>
              <p>
                Three Optuna-tuned (30 trials each) classifiers — Random Forest, XGBoost, LightGBM —
                each wrapped in CalibratedClassifierCV (isotonic regression, 5-fold) for reliable
                probability outputs. The best model by validation AUC is selected as the primary predictor.
              </p>
            </div>

            <div className="space-y-3">
              <h3 className="text-white font-medium">Explainability</h3>
              <ul className="space-y-1 list-disc list-inside">
                <li>SHAP TreeExplainer — feature attribution per patient</li>
                <li>LIME — local surrogate model explanation</li>
                <li>DiCE-ML — diverse counterfactual recourse (with linear fallback)</li>
              </ul>
            </div>

            <div className="space-y-3">
              <h3 className="text-white font-medium">Fairness</h3>
              <p>
                Fairlearn MetricFrame across gender (RIAGENDR) and ethnicity (RIDRETH3).
                Metrics: selection rate, true positive rate, false positive rate.
                Equal Opportunity Difference computed against the privileged group.
              </p>
            </div>

            <div className="space-y-3">
              <h3 className="text-white font-medium">Data</h3>
              <p>
                NHANES 2017–2018 (J cycle) — 5 SAS XPT files: DEMO_J, BMX_J, BPX_J, GLU_J, INS_J.
                Merged on SEQN, median-imputed missing values.
                Feature engineering: BP/Age ratio, BMI×Age, Insulin×BMI, Insulin/Glucose.
              </p>
            </div>

            <div className="glass p-4 border border-blue-500/15 bg-blue-500/5 text-xs text-blue-300">
              <strong>Research Paper:</strong> Target journals — IEEE JBHI, Computers in Biology and Medicine,
              Scientific Reports. Suggested title:{" "}
              <em>
                "Multi-Modal Explainable AI Framework for Metabolic Syndrome Risk Stratification
                Using NHANES Population Data with Fairness-Aware Clinical Decision Support"
              </em>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
