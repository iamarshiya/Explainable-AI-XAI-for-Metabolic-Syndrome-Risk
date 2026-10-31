"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, Stethoscope, LayoutDashboard } from "lucide-react";
import { cn } from "@/lib/utils";

const links = [
  { href: "/",          label: "Home",      icon: LayoutDashboard },
  { href: "/patient",   label: "Patient",   icon: Activity },
  { href: "/clinician", label: "Clinician", icon: Stethoscope },
];

export default function Navbar() {
  const path = usePathname();
  return (
    <nav className="fixed top-0 inset-x-0 z-50 h-16 border-b border-slate-800/60 bg-slate-950/80 backdrop-blur-xl">
      <div className="mx-auto max-w-7xl h-full px-6 flex items-center justify-between">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-2.5 group">
          <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center shadow-lg shadow-blue-500/30 group-hover:bg-blue-500 transition-colors">
            <Activity size={16} className="text-white" />
          </div>
          <span className="font-semibold text-white tracking-tight">
            MetaRisk <span className="text-blue-400">XAI</span>
          </span>
        </Link>

        {/* Nav links */}
        <div className="flex items-center gap-1">
          {links.map(({ href, label, icon: Icon }) => (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all",
                path === href
                  ? "bg-blue-600/20 text-blue-300 border border-blue-500/30"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
              )}
            >
              <Icon size={15} />
              {label}
            </Link>
          ))}
        </div>

        {/* Status badge */}
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          NHANES-J · v2.0
        </div>
      </div>
    </nav>
  );
}
