"use client";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, ResponsiveContainer, Area, AreaChart,
} from "recharts";
import type { SurvivalResult } from "@/lib/api";

interface Props { data: SurvivalResult }

export default function SurvivalCurve({ data }: Props) {
  const rows = data.years.map((y, i) => ({
    year: y,
    prob: Math.round(data.survival_probability[i] * 100),
  }));

  return (
    <div>
      <p className="text-xs text-slate-500 mb-4">
        Estimated probability of remaining MetS-free over a 10-year horizon
      </p>
      <ResponsiveContainer width="100%" height={280}>
        <AreaChart data={rows} margin={{ left: 8, right: 16, top: 8, bottom: 8 }}>
          <defs>
            <linearGradient id="survGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor="#14b8a6" stopOpacity={0.25} />
              <stop offset="95%" stopColor="#14b8a6" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis
            dataKey="year"
            label={{ value: "Years", position: "insideBottom", offset: -2, fill: "#64748b", fontSize: 12 }}
            tick={{ fill: "#94a3b8", fontSize: 11 }}
            axisLine={{ stroke: "#334155" }}
            tickLine={false}
          />
          <YAxis
            domain={[0, 100]}
            tickFormatter={(v) => `${v}%`}
            tick={{ fill: "#94a3b8", fontSize: 11 }}
            axisLine={{ stroke: "#334155" }}
            tickLine={false}
          />
          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const d = payload[0].payload as (typeof rows)[0];
              return (
                <div className="glass px-3 py-2 text-xs">
                  <p className="text-white font-semibold">Year {d.year}</p>
                  <p className="text-teal-400">MetS-free: {d.prob}%</p>
                </div>
              );
            }}
          />
          <ReferenceLine y={50} stroke="#f97316" strokeDasharray="4 2" strokeWidth={1} />
          <Area
            type="monotone"
            dataKey="prob"
            stroke="#14b8a6"
            strokeWidth={2.5}
            fill="url(#survGrad)"
            dot={false}
            activeDot={{ r: 5, fill: "#14b8a6", strokeWidth: 0 }}
          />
        </AreaChart>
      </ResponsiveContainer>
      <p className="text-xs text-slate-600 mt-2 text-center">
        Orange line = 50% survival threshold
      </p>
    </div>
  );
}
