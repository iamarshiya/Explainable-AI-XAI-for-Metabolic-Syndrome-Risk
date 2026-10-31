import Link from "next/link";
import { Activity, Stethoscope, BarChart2, Shield, Brain, ArrowRight, Zap } from "lucide-react";

const stats = [
  { label: "ROC-AUC",    value: "0.91",   sub: "LightGBM (calibrated)" },
  { label: "Accuracy",   value: "85%",    sub: "10-fold stratified CV" },
  { label: "F1 Score",   value: "0.83",   sub: "NHANES-J cohort" },
  { label: "Cohort",     value: "8 k+",   sub: "NHANES 2017–2018 subjects" },
];

const features = [
  {
    icon: Brain,
    title: "Multi-Model Ensemble",
    desc: "RF + XGBoost + LightGBM — each Optuna-tuned and isotonically calibrated for reliable probabilities.",
    color: "text-blue-400",
    bg: "bg-blue-500/10 border-blue-500/20",
  },
  {
    icon: Zap,
    title: "SHAP · LIME · DiCE",
    desc: "Waterfall plots, dependence plots, local LIME explanations, and diverse counterfactual recourse pathways.",
    color: "text-violet-400",
    bg: "bg-violet-500/10 border-violet-500/20",
  },
  {
    icon: Shield,
    title: "Fairness Auditing",
    desc: "Demographic parity, Equal Opportunity Difference across gender + 5 NHANES ethnicity groups.",
    color: "text-teal-400",
    bg: "bg-teal-500/10 border-teal-500/20",
  },
  {
    icon: BarChart2,
    title: "Survival Analysis",
    desc: "10-year metabolic risk horizon via Cox Proportional Hazards model with NHANES-derived hazard ratios.",
    color: "text-amber-400",
    bg: "bg-amber-500/10 border-amber-500/20",
  },
];

const portals = [
  {
    href: "/patient",
    icon: Activity,
    title: "Patient Portal",
    desc: "Enter vitals, get real-time risk score, SHAP waterfall, survival curve, and personalised recourse recommendations.",
    cta: "Open Patient Portal",
    gradient: "from-blue-600 to-blue-800",
    glow: "shadow-blue-500/30",
  },
  {
    href: "/clinician",
    icon: Stethoscope,
    title: "Clinician Dashboard",
    desc: "Model benchmarks, fairness audit across demographics, global SHAP importance, and batch patient analysis.",
    cta: "Open Clinician Dashboard",
    gradient: "from-teal-600 to-teal-800",
    glow: "shadow-teal-500/30",
  },
];

export default function Home() {
  return (
    <div className="min-h-screen">
      {/* Hero */}
      <section className="relative overflow-hidden">
        <div
          className="absolute inset-0"
          style={{
            background:
              "radial-gradient(ellipse 80% 60% at 50% -10%, rgba(59,130,246,0.18) 0%, transparent 70%)",
          }}
        />
        <div className="relative mx-auto max-w-5xl px-6 pt-24 pb-20 text-center">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-300 text-xs font-medium mb-8">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-400" />
            ATP-III Proxy · NHANES-J 2017–2018 · Conformal Prediction
          </div>

          <h1 className="text-5xl md:text-6xl font-bold tracking-tight text-white mb-6">
            Explainable AI for{" "}
            <span className="gradient-text">Metabolic Syndrome</span>
            <br />Risk Stratification
          </h1>

          <p className="max-w-2xl mx-auto text-lg text-slate-400 mb-10 leading-relaxed">
            Production-grade XAI platform built on NHANES population data. Multi-model
            calibrated ensemble with SHAP · LIME · DiCE explainability, demographic
            fairness auditing, and ADA-grounded clinical narratives.
          </p>

          <div className="flex flex-wrap justify-center gap-4">
            <Link
              href="/patient"
              className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold shadow-lg shadow-blue-500/30 transition-all hover:shadow-blue-500/50 hover:-translate-y-0.5"
            >
              Patient Portal <ArrowRight size={16} />
            </Link>
            <Link
              href="/clinician"
              className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold border border-slate-700 transition-all hover:-translate-y-0.5"
            >
              Clinician Dashboard <ArrowRight size={16} />
            </Link>
          </div>
        </div>
      </section>

      {/* Stats */}
      <section className="mx-auto max-w-5xl px-6 pb-16">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {stats.map((s) => (
            <div key={s.label} className="glass p-5 text-center">
              <div className="text-3xl font-bold text-white mb-1">{s.value}</div>
              <div className="text-sm font-semibold text-blue-300 mb-0.5">{s.label}</div>
              <div className="text-xs text-slate-500">{s.sub}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Features */}
      <section className="mx-auto max-w-5xl px-6 pb-16">
        <h2 className="text-2xl font-bold text-white mb-8 text-center">Platform Capabilities</h2>
        <div className="grid md:grid-cols-2 gap-4">
          {features.map((f) => (
            <div key={f.title} className={`glass p-6 border ${f.bg}`}>
              <div className={`w-10 h-10 rounded-lg bg-current/10 flex items-center justify-center mb-4 ${f.color}`}>
                <f.icon size={20} className={f.color} />
              </div>
              <h3 className="font-semibold text-white mb-2">{f.title}</h3>
              <p className="text-sm text-slate-400 leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Portals */}
      <section className="mx-auto max-w-5xl px-6 pb-24">
        <h2 className="text-2xl font-bold text-white mb-8 text-center">Choose Your Portal</h2>
        <div className="grid md:grid-cols-2 gap-6">
          {portals.map((p) => (
            <Link key={p.href} href={p.href} className="group glass p-8 hover:border-slate-600 transition-all hover:-translate-y-1">
              <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${p.gradient} flex items-center justify-center mb-5 shadow-lg ${p.glow} group-hover:scale-105 transition-transform`}>
                <p.icon size={22} className="text-white" />
              </div>
              <h3 className="text-xl font-bold text-white mb-3">{p.title}</h3>
              <p className="text-slate-400 text-sm leading-relaxed mb-5">{p.desc}</p>
              <span className="inline-flex items-center gap-1.5 text-sm font-medium text-blue-400 group-hover:text-blue-300 transition-colors">
                {p.cta} <ArrowRight size={14} className="group-hover:translate-x-1 transition-transform" />
              </span>
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}
