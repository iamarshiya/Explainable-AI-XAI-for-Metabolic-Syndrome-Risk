"use client";
import { useState, useRef } from "react";
import { Upload, Download, AlertCircle, CheckCircle } from "lucide-react";
import { batchPredict } from "@/lib/api";

type Row = Record<string, unknown>;

export default function BatchUpload() {
  const [rows, setRows]     = useState<Row[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError]   = useState("");
  const ref = useRef<HTMLInputElement>(null);

  async function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setError("");
    setLoading(true);
    try {
      const res = await batchPredict(file);
      setRows(res.predictions);
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? "Upload failed";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }

  function downloadCSV() {
    if (!rows.length) return;
    const cols = Object.keys(rows[0]);
    const csv  = [cols.join(","), ...rows.map((r) => cols.map((c) => r[c]).join(","))].join("\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    a.download = "metarisk_batch_results.csv";
    a.click();
  }

  const template = "Age,BMI,BloodPressure,Insulin,Glucose\n45,28.5,135,22,105\n60,31.2,145,30,118";

  return (
    <div className="space-y-6">
      {/* Instructions */}
      <div className="glass p-4 border border-blue-500/15 bg-blue-500/5">
        <p className="text-sm text-blue-300 font-medium mb-2">Required CSV columns:</p>
        <p className="text-xs text-slate-400 font-mono">Age, BMI, BloodPressure, Insulin, Glucose (optional)</p>
        <button
          onClick={() => {
            const a = document.createElement("a");
            a.href = URL.createObjectURL(new Blob([template], { type: "text/csv" }));
            a.download = "batch_template.csv";
            a.click();
          }}
          className="mt-3 text-xs text-blue-400 hover:text-blue-300 underline"
        >
          Download template CSV
        </button>
      </div>

      {/* Drop zone */}
      <div
        className="border-2 border-dashed border-slate-700 rounded-xl p-10 text-center hover:border-blue-500/50 hover:bg-blue-500/5 transition-all cursor-pointer"
        onClick={() => ref.current?.click()}
      >
        <Upload size={32} className="mx-auto text-slate-600 mb-3" />
        <p className="text-slate-400 font-medium">Click to upload or drag & drop</p>
        <p className="text-xs text-slate-600 mt-1">CSV files only</p>
        <input ref={ref} type="file" accept=".csv" className="hidden" onChange={handleFile} />
      </div>

      {/* Feedback */}
      {loading && (
        <div className="flex items-center gap-2 text-sm text-blue-400">
          <span className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
          Scoring patients…
        </div>
      )}
      {error && (
        <div className="flex items-center gap-2 text-sm text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg p-3">
          <AlertCircle size={15} />
          {error}
        </div>
      )}

      {/* Results */}
      {rows.length > 0 && (
        <div>
          <div className="flex items-center justify-between mb-3">
            <p className="text-sm text-slate-400 flex items-center gap-2">
              <CheckCircle size={15} className="text-emerald-400" />
              {rows.length} patients scored
            </p>
            <button
              onClick={downloadCSV}
              className="flex items-center gap-2 text-xs px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 transition-colors"
            >
              <Download size={13} /> Download results
            </button>
          </div>
          <div className="overflow-x-auto max-h-72 overflow-y-auto rounded-lg border border-slate-800">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-slate-900">
                <tr>
                  {Object.keys(rows[0]).map((k) => (
                    <th key={k} className="text-left px-3 py-2.5 text-slate-500 font-semibold border-b border-slate-800 whitespace-nowrap">
                      {k}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => {
                  const risk = r.Risk_Label as string;
                  return (
                    <tr key={i} className="border-b border-slate-800/50 hover:bg-slate-800/20">
                      {Object.entries(r).map(([k, v]) => (
                        <td key={k} className="px-3 py-2 whitespace-nowrap">
                          {k === "Risk_Label" ? (
                            <span
                              className={`px-2 py-0.5 rounded-full font-medium text-xs
                                ${risk === "High Risk"     ? "bg-red-500/15 text-red-400"
                                : risk === "Moderate Risk" ? "bg-orange-500/15 text-orange-400"
                                :                           "bg-green-500/15 text-green-400"}`}
                            >
                              {v as string}
                            </span>
                          ) : (
                            <span className="text-slate-300 font-mono">
                              {typeof v === "number" ? v.toFixed(4) : String(v)}
                            </span>
                          )}
                        </td>
                      ))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
