"use client";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from "recharts";
import type { BenchmarkRow } from "@/lib/api";

const COLORS = ["#64748b", "#3b82f6", "#8b5cf6", "#f97316", "#14b8a6"];
const BEST   = "LightGBM";

function parseAuc(s: string) {
  return parseFloat(s.split("±")[0].trim());
}

interface Props { rows: BenchmarkRow[] }

export default function BenchmarkTable({ rows }: Props) {
  const chartData = rows.map((r, i) => ({
    name:  r.Model.replace(" (", "\n("),
    auc:   parseAuc(r["ROC-AUC"]),
    color: COLORS[i % COLORS.length],
  }));

  return (
    <div className="space-y-6">
      {/* Bar chart */}
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={chartData} margin={{ left: 0, right: 16, top: 8, bottom: 40 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis
            dataKey="name"
            tick={{ fill: "#94a3b8", fontSize: 11 }}
            axisLine={{ stroke: "#334155" }}
            tickLine={false}
            interval={0}
            angle={-15}
            textAnchor="end"
          />
          <YAxis
            domain={[0.6, 1]}
            tickFormatter={(v) => v.toFixed(2)}
            tick={{ fill: "#94a3b8", fontSize: 11 }}
            axisLine={{ stroke: "#334155" }}
            tickLine={false}
          />
          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              return (
                <div className="glass px-3 py-2 text-xs">
                  <p className="text-white font-semibold">{payload[0].payload.name}</p>
                  <p className="text-blue-300">ROC-AUC: {payload[0].value}</p>
                </div>
              );
            }}
          />
          <Bar dataKey="auc" radius={[4,4,0,0]}>
            {chartData.map((d, i) => (
              <Cell key={i} fill={d.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-800">
              {["Model", "Accuracy", "ROC-AUC", "F1-Score"].map((h) => (
                <th key={h} className="text-left px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wider">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const isBest = r.Model === BEST;
              return (
                <tr
                  key={r.Model}
                  className={`border-b border-slate-800/50 transition-colors hover:bg-slate-800/30
                    ${isBest ? "bg-teal-500/5" : ""}`}
                >
                  <td className="px-4 py-3 font-medium text-white flex items-center gap-2">
                    {r.Model}
                    {isBest && (
                      <span className="text-xs px-2 py-0.5 rounded-full bg-teal-500/15 text-teal-400 border border-teal-500/25">
                        Best
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-slate-300 font-mono text-xs">{r.Accuracy}</td>
                  <td className="px-4 py-3 text-slate-300 font-mono text-xs">{r["ROC-AUC"]}</td>
                  <td className="px-4 py-3 text-slate-300 font-mono text-xs">{r["F1-Score"]}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
