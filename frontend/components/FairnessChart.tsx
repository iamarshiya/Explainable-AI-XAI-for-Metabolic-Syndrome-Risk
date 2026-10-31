"use client";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer,
} from "recharts";
import type { FairnessResult } from "@/lib/api";

interface Props { data: FairnessResult }

function Group({ title, rows }: { title: string; rows: FairnessResult["gender"] }) {
  const chartData = rows.map((r) => ({
    name:           r.group,
    "Selection Rate": Math.round(r.selection_rate * 100),
    "TPR":           Math.round(r.tpr * 100),
    "FPR":           Math.round(r.fpr * 100),
  }));

  return (
    <div>
      <h3 className="font-semibold text-white mb-4">{title}</h3>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={chartData} margin={{ left: 0, right: 16, top: 8, bottom: 24 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis
            dataKey="name"
            tick={{ fill: "#94a3b8", fontSize: 11 }}
            axisLine={{ stroke: "#334155" }}
            tickLine={false}
            angle={rows.length > 3 ? -20 : 0}
            textAnchor={rows.length > 3 ? "end" : "middle"}
          />
          <YAxis
            tickFormatter={(v) => `${v}%`}
            tick={{ fill: "#94a3b8", fontSize: 11 }}
            axisLine={{ stroke: "#334155" }}
            tickLine={false}
            domain={[0, 100]}
          />
          <Tooltip
            content={({ active, payload, label }) => {
              if (!active || !payload?.length) return null;
              return (
                <div className="glass px-3 py-2 text-xs space-y-1">
                  <p className="text-white font-semibold mb-1">{label}</p>
                  {payload.map((p) => (
                    <p key={p.name} style={{ color: p.color }}>
                      {p.name}: {p.value}%
                    </p>
                  ))}
                </div>
              );
            }}
          />
          <Legend wrapperStyle={{ color: "#94a3b8", fontSize: 12 }} />
          <Bar dataKey="Selection Rate" fill="#3b82f6" radius={[3,3,0,0]} />
          <Bar dataKey="TPR"            fill="#22c55e" radius={[3,3,0,0]} />
          <Bar dataKey="FPR"            fill="#ef4444" radius={[3,3,0,0]} />
        </BarChart>
      </ResponsiveContainer>

      {/* EOD for gender */}
      {rows.some((r) => r.eod !== undefined) && (
        <div className="mt-4 grid grid-cols-2 gap-3">
          {rows.map((r) => (
            <div key={r.group} className="glass p-3">
              <div className="text-xs text-slate-500 mb-1">{r.group}</div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-slate-400">Equal Opp. Diff</span>
                <span
                  className={`text-sm font-mono font-bold ${
                    Math.abs(r.eod ?? 0) < 0.05 ? "text-green-400"
                    : Math.abs(r.eod ?? 0) < 0.10 ? "text-amber-400"
                    : "text-red-400"
                  }`}
                >
                  {r.eod !== undefined ? (r.eod >= 0 ? "+" : "") + r.eod.toFixed(3) : "—"}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function FairnessChart({ data }: Props) {
  return (
    <div className="space-y-8">
      <Group title="By Gender" rows={data.gender} />
      <div className="border-t border-slate-800" />
      <Group title="By Ethnicity (NHANES RIDRETH3)" rows={data.ethnicity} />
    </div>
  );
}
