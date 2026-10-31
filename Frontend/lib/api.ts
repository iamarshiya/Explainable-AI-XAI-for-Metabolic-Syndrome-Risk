import axios from "axios";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export const api = axios.create({ baseURL: BASE });

// ── Types ─────────────────────────────────────────────────────────────────────
export interface PatientInput {
  age: number;
  bmi: number;
  blood_pressure: number;
  insulin: number;
  glucose: number;
  waist?: number | null; // cm — server estimates from BMI if omitted
}

export interface PredictResult {
  risk_probability: number;
  risk_label: "High Risk" | "Moderate Risk" | "Low Risk";
  model_scores: Record<string, number>;
  waist_used?: number;
}

export interface ShapResult {
  features: string[];
  shap_values: number[];
  base_value: number;
  feature_values: number[];
}

export interface SurvivalResult {
  years: number[];
  survival_probability: number[];
}

export interface RecourseItem {
  feature: string;
  current: number;
  target: number;
  change: number;
  action: string;
}

export interface FairnessRow {
  group: string;
  selection_rate: number;
  tpr: number;
  fpr: number;
  eod?: number;
}

export interface FairnessResult {
  gender: FairnessRow[];
  ethnicity: FairnessRow[];
}

export interface BenchmarkRow {
  Model: string;
  Accuracy: string;
  "ROC-AUC": string;
  "F1-Score": string;
}

// ── API calls ─────────────────────────────────────────────────────────────────
export const predictRisk = (p: PatientInput) =>
  api.post<PredictResult>("/predict", p).then((r) => r.data);

export const explainShap = (p: PatientInput) =>
  api.post<ShapResult>("/shap", p).then((r) => r.data);

export const getSurvival = (p: { age: number; bmi: number; blood_pressure: number }) =>
  api.post<SurvivalResult>("/survival", p).then((r) => r.data);

export const getRecourse = (p: PatientInput) =>
  api.post<{ suggestions: RecourseItem[] }>("/recourse", p).then((r) => r.data);

export const getFairness = () =>
  api.get<FairnessResult>("/fairness").then((r) => r.data);

export const getBenchmark = () =>
  api.get<BenchmarkRow[]>("/benchmark").then((r) => r.data);

export const batchPredict = (file: File) => {
  const fd = new FormData();
  fd.append("file", file);
  return api
    .post<{ predictions: Record<string, unknown>[]; count: number }>("/batch", fd)
    .then((r) => r.data);
};