"use client";
import { TrendingDown, TrendingUp, ArrowRight } from "lucide-react";
import type { RecourseItem } from "@/lib/api";

interface Props { items: RecourseItem[] }

export default function RecourseCard({ items }: Props) {
  if (!items.length) {
    return (
      <div className="text-center py-10 text-slate-500">
        <p className="text-2xl mb-2">🎉</p>
        <p className="font-medium text-slate-400">All vitals within healthy ranges!</p>
        <p className="text-sm mt-1">No recourse actions recommended at this time.</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {items.map((item) => {
        const isDown = item.change < 0;
        return (
          <div key={item.feature} className="glass p-5 border border-amber-500/10 hover:border-amber-500/25 transition-colors">
            <div className="flex items-start justify-between gap-4">
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-2">
                  <span className="font-semibold text-white">{item.feature}</span>
                  {isDown ? (
                    <TrendingDown size={15} className="text-green-400" />
                  ) : (
                    <TrendingUp size={15} className="text-red-400" />
                  )}
                </div>
                <p className="text-sm text-slate-400 leading-relaxed">{item.action}</p>
              </div>

              {/* Current → Target */}
              <div className="flex items-center gap-2 shrink-0 bg-slate-800/60 rounded-lg px-4 py-2 border border-slate-700/40">
                <div className="text-center">
                  <div className="text-xs text-slate-500 mb-0.5">Current</div>
                  <div className="text-lg font-bold text-white font-mono">{item.current}</div>
                </div>
                <ArrowRight size={14} className="text-slate-600" />
                <div className="text-center">
                  <div className="text-xs text-slate-500 mb-0.5">Target</div>
                  <div className="text-lg font-bold text-emerald-400 font-mono">{item.target}</div>
                </div>
              </div>
            </div>

            {/* Change pill */}
            <div className="mt-3">
              <span
                className={`inline-flex items-center gap-1 text-xs px-2.5 py-1 rounded-full font-medium
                  ${isDown ? "bg-green-500/10 text-green-400 border border-green-500/20"
                           : "bg-red-500/10 text-red-400 border border-red-500/20"}`}
              >
                {item.change >= 0 ? "+" : ""}{item.change} required change
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
