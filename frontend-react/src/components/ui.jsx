export function Card({ title, children, right }) {
  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-5">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-lg font-semibold text-slate-900">{title}</h2>
        {right || null}
      </div>
      {children}
    </div>
  );
}

export function Button({ children, variant = "primary", ...props }) {
  const base = "px-4 py-2 rounded-lg text-sm font-semibold transition";
  const styles =
    variant === "primary"
      ? "bg-blue-600 hover:bg-blue-700 text-white"
      : variant === "ghost"
      ? "bg-transparent hover:bg-slate-100 text-slate-800"
      : "bg-slate-200 hover:bg-slate-300 text-slate-900";

  return (
    <button className={`${base} ${styles}`} {...props}>
      {children}
    </button>
  );
}

export function Badge({ children, tone = "neutral" }) {
  const cls =
    tone === "good"
      ? "bg-emerald-100 text-emerald-800"
      : tone === "bad"
      ? "bg-rose-100 text-rose-800"
      : "bg-slate-100 text-slate-800";

  return <span className={`px-2 py-1 rounded-full text-xs font-semibold ${cls}`}>{children}</span>;
}

export function Progress({ value }) {
  const pct = Math.max(0, Math.min(100, value));
  return (
    <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
      <div className="h-2 bg-blue-600" style={{ width: `${pct}%` }} />
    </div>
  );
}
