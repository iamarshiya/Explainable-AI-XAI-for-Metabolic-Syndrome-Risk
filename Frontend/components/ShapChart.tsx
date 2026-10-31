"use client";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, Cell, ResponsiveContainer,
} from "recharts";
import type { ShapResult } from "@/lib/api";

const LABEL_MAP: Record<string, string> = {
  Age:                    "Age",
  BMI:                    "BMI",
  BloodPressure:          "Blood Pressure",
  Insulin:                "Insulin",
  Glucose:                "Glucose",
  WaistCircumference:     "Waist Circum.",
  BP_Age_ratio:           "BP/Age Ratio",
  BMI_Age_interaction:    "BMI × Age",
  Insulin_BMI_interaction:"Insulin × BMI",
  Insulin_Glucose_ratio:  "Insulin/Glucose",
};

interface Props { data: ShapResult }

export default function ShapChart({ data }: Props) {
  const rows = data.features
    .map((f, i) => ({
      name:  LABEL_MAP[f] ?? f,
      value: data.shap_values[i],
      raw:   data.feature_values[i],
    }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

  return (
    <div>
      <p className="text-xs text-slate-500 mb-4">
        Base value: <span className="text-slate-300 font-mono">{data.base_value.toFixed(3)}</span>
        &nbsp;· Positive = increases risk · Negative = decreases risk
      </p>
      <ResponsiveContainer width="100%" height={320}>
        <BarChart data={rows} layout="vertical" margin={{ left: 24, right: 24, top: 8, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
          <XAxis
            type="number"
            tick={{ fill: "#94a3b8", fontSize: 11 }}
            axisLine={{ stroke: "#334155" }}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="name"
            width={120}
            tick={{ fill: "#94a3b8", fontSize: 12 }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            cursor={{ fill: "rgba(255,255,255,0.03)" }}
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const d = payload[0].payload as (typeof rows)[0];
              return (
                <div className="glass px-3 py-2 text-xs">
                  <p className="text-white font-semibold mb-1">{d.name}</p>
                  <p className="text-slate-400">
                    Raw value: <span className="text-slate-200 font-mono">{d.raw.toFixed(2)}</span>
                  </p>
                  <p className="text-slate-400">
                    SHAP: <span className={d.value >= 0 ? "text-red-400" : "text-green-400"} style={{fontFamily:"monospace"}}>
                      {d.value >= 0 ? "+" : ""}{d.value.toFixed(4)}
                    </span>
                  </p>
                </div>
              );
            }}
          />
          <ReferenceLine x={0} stroke="#475569" strokeWidth={1} />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {rows.map((r, i) => (
              <Cell key={i} fill={r.value >= 0 ? "#ef4444" : "#22c55e"} fillOpacity={0.8} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
