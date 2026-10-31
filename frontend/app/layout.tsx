import type { Metadata } from "next";
import "./globals.css";
import Navbar from "@/components/Navbar";

export const metadata: Metadata = {
  title: "MetaRisk XAI — Metabolic Syndrome Risk Platform",
  description:
    "Production-grade XAI platform for metabolic syndrome risk stratification using NHANES data, multi-model ensemble, SHAP/LIME explainability, and fairness auditing.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-950 text-slate-200 antialiased">
        <Navbar />
        <main className="pt-16">{children}</main>
      </body>
    </html>
  );
}
