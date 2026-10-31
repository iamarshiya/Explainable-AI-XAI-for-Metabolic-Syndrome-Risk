"use client";
import { PieChart, Pie, Cell, ResponsiveContainer } from "recharts";

interface Props {
  probability: number; // 0..1
  label: string;
}

export default function RiskGauge({ probability, label }: Props) {
  const pct = Math.round(probability * 100);

  const color =
    probability >= 0.7 ? "#ef4444" :
    probability >= 0.4 ? "#f97316" :
    "#22c55e";

  const data = [
    { value: pct,       fill: color },
    { value: 100 - pct, fill: "#1e293b" },
  ];

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-48 h-28">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="90%"
              startAngle={180}
              endAngle={0}
              innerRadius={55}
              outerRadius={75}
              dataKey="value"
              strokeWidth={0}
            >
              {data.map((entry, i) => (
                <Cell key={i} fill={entry.fill} />
              ))}
            </Pie>
          </PieChart>
        </ResponsiveContainer>

        {/* Centre readout */}
        <div className="absolute inset-x-0 bottom-0 flex flex-col items-center">
          <span className="text-4xl font-bold text-white leading-none">
            {pct}
            <span className="text-xl text-slate-400">%</span>
          </span>
        </div>
      </div>

      {/* Label badge */}
      <span
        className="mt-3 px-4 py-1.5 rounded-full text-sm font-semibold"
        style={{
          background: `${color}22`,
          color,
          border: `1px solid ${color}44`,
        }}
      >
        {label}
      </span>
    </div>
  );
}
