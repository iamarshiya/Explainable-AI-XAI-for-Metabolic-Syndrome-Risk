"use client";
import { useState } from "react";
import {
  Activity, Zap, TrendingUp, RotateCcw, Upload, ChevronRight,
  Loader2, AlertTriangle,
} from "lucide-react";
import {
  predictRisk, explainShap, getSurvival, getRecourse,
  type PatientInput, type PredictResult, type ShapResult,
  type SurvivalResult, type RecourseItem,
} from "@/lib/api";
import RiskGauge    from "@/components/RiskGauge";
import ShapChart    from "@/components/ShapChart";
import SurvivalCurve from "@/components/SurvivalCurve";
import RecourseCard from "@/components/RecourseCard";
import BatchUpload  from "@/components/BatchUpload";

// ── Slider field ───────────────────────────────────────────────────────────────
function Slider({
  label, unit, min, max, step = 0.1, value, onChange,
}: {
  label: string; unit: string; min: number; max: number;
  step?: number; value: number; onChange: (v: number) => void;
}) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <label className="text-xs font-medium text-slate-400">{label}</label>
        <span className="text-sm font-semibold text-white font-mono">
          {value.toFixed(1)} <span className="text-slate-500 text-xs">{unit}</span>
        </span>
      </div>
      <input
        type="range"
        min={min} max={max} step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full"
      />
      <div className="flex justify-between text-xs text-slate-600 mt-0.5">
        <span>{min}</span><span>{max}</span>
      </div>
    </div>
  );
}

// ── Tab button ─────────────────────────────────────────────────────────────────
function Tab({ active, onClick, icon: Icon, label }: {
  active: boolean; onClick: () => void;
  icon: React.ElementType; label: string;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium rounded-lg transition-all
        ${active
          ? "bg-blue-600/20 text-blue-300 border border-blue-500/30"
          : "text-slate-500 hover:text-slate-300 hover:bg-slate-800/60"
        }`}
    >
      <Icon size={14} />
      {label}
    </button>
  );
}

// ── Model score badges ─────────────────────────────────────────────────────────
function ModelScores({ scores }: { scores: Record<string, number> }) {
  const labels: Record<string, string> = { rf: "RF", xgb: "XGB", lgb: "LGB", best: "Best" };
  return (
    <div className="flex flex-wrap gap-2 mt-4">
      {Object.entries(scores).map(([k, v]) => (
        <div key={k} className="glass px-3 py-1.5 flex items-center gap-2">
          <span className="text-xs text-slate-500">{labels[k] ?? k}</span>
          <span className={`text-sm font-bold font-mono
            ${v >= 0.7 ? "text-red-400" : v >= 0.4 ? "text-orange-400" : "text-green-400"}`}>
            {(v * 100).toFixed(1)}%
          </span>
        </div>
      ))}
    </div>
  );
}

type TabId = "diagnostic" | "shap" | "survival" | "recourse" | "batch";

export default function PatientPortal() {
  // Input state
  const [age, setAge]   = useState(45);
  const [bmi, setBmi]   = useState(27.5);
  const [bp,  setBp]    = useState(130);
  const [ins, setIns]   = useState(18);
  const [glu, setGlu]   = useState(105);
  const [autoWaist, setAutoWaist] = useState(true);
  const [waist, setWaist] = useState(92);

  // Results
  const [predict, setPredict]   = useState<PredictResult | null>(null);
  const [shap,    setShap]      = useState<ShapResult | null>(null);
  const [survival,setSurvival]  = useState<SurvivalResult | null>(null);
  const [recourse,setRecourse]  = useState<RecourseItem[] | null>(null);

  const [tab,     setTab]   = useState<TabId>("diagnostic");
  const [loading, setLoading] = useState(false);
  const [error,   setError] = useState("");

  const input: PatientInput = {
    age, bmi, blood_pressure: bp, insulin: ins, glucose: glu,
    waist: autoWaist ? null : waist,
  };

  async function runAnalysis() {
    setLoading(true);
    setError("");
    try {
      const [pred, shapRes, survRes, recRes] = await Promise.all([
        predictRisk(input),
        explainShap(input),
        getSurvival({ age, bmi, blood_pressure: bp }),
        getRecourse(input),
      ]);
      setPredict(pred);
      setShap(shapRes);
      setSurvival(survRes);
      setRecourse(recRes.suggestions);
    } catch {
      setError("Could not reach the API. Make sure the FastAPI server is running.");
    } finally {
      setLoading(false);
    }
  }

  const tabs: { id: TabId; icon: React.ElementType; label: string }[] = [
    { id: "diagnostic", icon: Activity,   label: "Diagnostic"   },
    { id: "shap",       icon: Zap,        label: "SHAP Waterfall"},
    { id: "survival",   icon: TrendingUp, label: "Survival"     },
    { id: "recourse",   icon: RotateCcw,  label: "Recourse"     },
    { id: "batch",      icon: Upload,     label: "Batch Upload" },
  ];

  return (
    <div className="mx-auto max-w-6xl px-6 py-10">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">Patient Risk Portal</h1>
        <p className="text-slate-400">Enter vitals below and run the full XAI analysis.</p>
      </div>

      <div className="grid lg:grid-cols-[320px_1fr] gap-6">
        {/* ── Left panel: inputs ─────────────────────────────────────── */}
        <div className="glass p-6 space-y-5 h-fit">
          <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">Patient Vitals</h2>

          <Slider label="Age"            unit="yrs"   min={18}  max={85}  step={1}   value={age} onChange={setAge} />
          <Slider label="BMI"            unit="kg/m²" min={15}  max={50}  step={0.1} value={bmi} onChange={setBmi} />
          <Slider label="Systolic BP"    unit="mmHg"  min={80}  max={200} step={1}   value={bp}  onChange={setBp}  />
          <Slider label="Insulin"        unit="µU/mL" min={2}   max={80}  step={0.5} value={ins} onChange={setIns} />
          <Slider label="Fasting Glucose" unit="mg/dL" min={60} max={200} step={1}   value={glu} onChange={setGlu} />

          {/* Waist — auto-estimated from BMI unless overridden */}
          <div className="pt-2 border-t border-slate-800">
            <label className="flex items-center gap-2 mb-3 cursor-pointer">
              <input
                type="checkbox"
                checked={autoWaist}
                onChange={(e) => setAutoWaist(e.target.checked)}
                className="w-3.5 h-3.5 rounded accent-blue-500"
              />
              <span className="text-xs text-slate-400">
                Estimate waist from BMI
                {autoWaist && (
                  <span className="text-slate-500 font-mono ml-1">
                    ≈ {(2.6 * bmi + 21).toFixed(1)} cm
                  </span>
                )}
              </span>
            </label>
            {!autoWaist && (
              <Slider
                label="Waist Circumference" unit="cm"
                min={55} max={160} step={0.5}
                value={waist} onChange={setWaist}
              />
            )}
          </div>

          <button
            onClick={runAnalysis}
            disabled={loading}
            className="w-full flex items-center justify-center gap-2 py-3 px-4 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-60 text-white font-semibold shadow-lg shadow-blue-500/25 transition-all hover:-translate-y-0.5 active:translate-y-0"
          >
            {loading ? <Loader2 size={16} className="animate-spin" /> : <ChevronRight size={16} />}
            {loading ? "Analysing…" : "Run Analysis"}
          </button>

          {error && (
            <div className="flex items-start gap-2 text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg p-3">
              <AlertTriangle size={14} className="mt-0.5 shrink-0" />
              {error}
            </div>
          )}
        </div>

        {/* ── Right panel: tabs ──────────────────────────────────────── */}
        <div className="space-y-4">
          {/* Tab bar */}
          <div className="flex flex-wrap gap-2">
            {tabs.map((t) => (
              <Tab key={t.id} active={tab === t.id} onClick={() => setTab(t.id)}
                   icon={t.icon} label={t.label} />
            ))}
          </div>

          {/* Tab content */}
          <div className="glass p-6 min-h-[420px]">

            {/* ─ Diagnostic ─ */}
            {tab === "diagnostic" && (
              <div>
                <h2 className="text-lg font-semibold text-white mb-6">Risk Assessment</h2>
                {predict ? (
                  <div className="space-y-6">
                    <div className="flex flex-col sm:flex-row items-start sm:items-center gap-8">
                      <RiskGauge
                        probability={predict.risk_probability}
                        label={predict.risk_label}
                      />
                      <div className="flex-1">
                        <p className="text-sm text-slate-400 mb-2">
                          MetS risk probability:{" "}
                          <span className="text-white font-bold text-lg font-mono">
                            {(predict.risk_probability * 100).toFixed(1)}%
                          </span>
                        </p>
                        <p className="text-xs text-slate-500 leading-relaxed">
                          Based on ATP-III proxy criteria applied to your vitals using
                          a calibrated multi-model ensemble (RF + XGBoost + LightGBM)
                          trained on NHANES 2017–2018 data.
                        </p>
                        <ModelScores scores={predict.model_scores} />
                      </div>
                    </div>

                    {/* Risk interpretation */}
                    <div className={`rounded-xl p-4 border text-sm
                      ${predict.risk_probability >= 0.7
                        ? "bg-red-500/10 border-red-500/20 text-red-300"
                        : predict.risk_probability >= 0.4
                        ? "bg-orange-500/10 border-orange-500/20 text-orange-300"
                        : "bg-green-500/10 border-green-500/20 text-green-300"}`}
                    >
                      {predict.risk_probability >= 0.7
                        ? "⚠ High risk — consider immediate clinical evaluation and lifestyle intervention."
                        : predict.risk_probability >= 0.4
                        ? "⚡ Moderate risk — preventive lifestyle changes recommended. Re-assess in 6 months."
                        : "✓ Low risk — maintain current healthy lifestyle and monitor annually."}
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center h-64 text-slate-600">
                    <Activity size={48} strokeWidth={1} className="mb-4" />
                    <p>Enter vitals and click <span className="text-slate-400">Run Analysis</span></p>
                  </div>
                )}
              </div>
            )}

            {/* ─ SHAP ─ */}
            {tab === "shap" && (
              <div>
                <h2 className="text-lg font-semibold text-white mb-2">SHAP Feature Attribution</h2>
                {shap ? (
                  <ShapChart data={shap} />
                ) : (
                  <div className="flex flex-col items-center justify-center h-64 text-slate-600">
                    <Zap size={48} strokeWidth={1} className="mb-4" />
                    <p>Run analysis to see SHAP waterfall</p>
                  </div>
                )}
              </div>
            )}

            {/* ─ Survival ─ */}
            {tab === "survival" && (
              <div>
                <h2 className="text-lg font-semibold text-white mb-2">10-Year Survival Curve</h2>
                {survival ? (
                  <SurvivalCurve data={survival} />
                ) : (
                  <div className="flex flex-col items-center justify-center h-64 text-slate-600">
                    <TrendingUp size={48} strokeWidth={1} className="mb-4" />
                    <p>Run analysis to see survival curve</p>
                  </div>
                )}
              </div>
            )}

            {/* ─ Recourse ─ */}
            {tab === "recourse" && (
              <div>
                <h2 className="text-lg font-semibold text-white mb-2">Actionable Recourse</h2>
                <p className="text-xs text-slate-500 mb-4">
                  Minimum changes needed to shift toward low-risk status.
                </p>
                {recourse !== null ? (
                  <RecourseCard items={recourse} />
                ) : (
                  <div className="flex flex-col items-center justify-center h-64 text-slate-600">
                    <RotateCcw size={48} strokeWidth={1} className="mb-4" />
                    <p>Run analysis to see recourse recommendations</p>
                  </div>
                )}
              </div>
            )}

            {/* ─ Batch ─ */}
            {tab === "batch" && (
              <div>
                <h2 className="text-lg font-semibold text-white mb-2">Batch Patient Scoring</h2>
                <BatchUpload />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
