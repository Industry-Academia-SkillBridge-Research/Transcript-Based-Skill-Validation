import { useEffect, useMemo, useState } from "react";
import { getSkillEvidence, getRoleEvidence } from "../api";

export default function XaiPanel({ open, onClose, studentId, selectedRole }) {
  const [skills, setSkills] = useState(null);
  const [roles, setRoles] = useState(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    if (!open) return;

    let alive = true;
    setErr("");
    setSkills(null);
    setRoles(null);

    (async () => {
      try {
        const s = await getSkillEvidence(studentId);
        if (!alive) return;
        setSkills(s);

        const r = await getRoleEvidence(studentId, selectedRole);
        if (!alive) return;
        setRoles(r);
      } catch (e) {
        if (!alive) return;
        setErr(e?.detail ? JSON.stringify(e.detail) : String(e));
      }
    })();

    return () => {
      alive = false;
    };
  }, [open, studentId, selectedRole]);

  const roleTitle = useMemo(() => roles?.role_name || selectedRole || "Top role", [roles, selectedRole]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50">
      <div className="absolute inset-0 bg-black/40" onClick={onClose} />
      <div className="absolute right-0 top-0 h-full w-full max-w-2xl bg-white shadow-xl overflow-y-auto">
        <div className="p-4 border-b flex items-center justify-between">
          <div>
            <div className="text-lg font-semibold">Explainability (XAI)</div>
            <div className="text-sm text-slate-500">Student: {studentId}</div>
          </div>
          <button className="px-3 py-2 rounded-lg border" onClick={onClose}>Close</button>
        </div>

        <div className="p-4 space-y-4">
          {err && (
            <div className="rounded-xl border border-red-200 bg-red-50 p-3 text-sm text-red-800">
              {err}
            </div>
          )}

          <section className="rounded-xl border p-4">
            <div className="font-semibold mb-2">Skill evidence</div>
            {!skills ? (
              <div className="text-sm text-slate-500">Loading…</div>
            ) : (
              <div className="space-y-3">
                {(skills.skills || []).map((s) => (
                  <div key={s.skill} className="rounded-lg bg-slate-50 p-3">
                    <div className="flex items-center justify-between">
                      <div className="font-medium">{s.skill}</div>
                      <div className="text-xs text-slate-600">
                        {s.level ? <span className="mr-2">{s.level}</span> : null}
                        {typeof s.score === "number" ? <span>score {s.score.toFixed(3)}</span> : null}
                      </div>
                    </div>
                    <div className="mt-2 text-sm text-slate-700 whitespace-pre-wrap">
                      {s.evidence ? s.evidence : "No evidence text available in CSV."}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>

          <section className="rounded-xl border p-4">
            <div className="font-semibold mb-2">Role evidence</div>
            <div className="text-sm text-slate-500 mb-3">Role: {roleTitle}</div>

            {!roles ? (
              <div className="text-sm text-slate-500">Loading…</div>
            ) : (
              <>
                <div className="text-sm mb-3">
                  Weak or missing: <span className="font-semibold">{roles.num_weak_or_missing}</span> / {roles.num_required_skills}
                </div>

                <div className="space-y-2">
                  {(roles.required_skills || []).slice(0, 25).map((r) => (
                    <div key={r.skill} className="flex items-start justify-between gap-3 rounded-lg bg-slate-50 p-3">
                      <div>
                        <div className="font-medium">{r.skill}</div>
                        <div className="text-xs text-slate-600">
                          {r.student_level || ""}
                        </div>
                      </div>
                      <div className="text-right text-xs text-slate-700">
                        {typeof r.required_importance === "number" && <div>req {r.required_importance.toFixed(2)}</div>}
                        {typeof r.student_score === "number" && <div>you {r.student_score.toFixed(3)}</div>}
                        {r.is_weak_or_missing ? (
                          <div className="mt-1 inline-flex rounded-full bg-red-100 px-2 py-0.5 text-red-700">gap</div>
                        ) : (
                          <div className="mt-1 inline-flex rounded-full bg-green-100 px-2 py-0.5 text-green-700">ok</div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>

                <div className="text-xs text-slate-500 mt-3">
                  Showing first 25 skills. You can paginate later.
                </div>
              </>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}
