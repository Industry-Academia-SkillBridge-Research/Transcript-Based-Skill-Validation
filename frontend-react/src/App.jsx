import React, { useMemo, useState } from "react";
import {
  getSkills,
  getRoles,
  prepareQuiz,
  submitQuiz,
  uploadTranscript,
  getSkillEvidence,
  getRoleEvidence,
} from "./api";

function classNames(...xs) {
  return xs.filter(Boolean).join(" ");
}

function Badge({ children, tone = "slate" }) {
  const map = {
    slate: "bg-slate-100 text-slate-700",
    blue: "bg-blue-100 text-blue-700",
    green: "bg-green-100 text-green-700",
    red: "bg-red-100 text-red-700",
    amber: "bg-amber-100 text-amber-800",
  };
  return (
    <span className={classNames("px-2 py-0.5 rounded-full text-xs font-medium", map[tone] || map.slate)}>
      {children}
    </span>
  );
}

function Card({ title, subtitle, right, children }) {
  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200">
      <div className="px-5 py-4 border-b border-slate-100 flex items-start justify-between gap-4">
        <div>
          <div className="text-slate-900 font-semibold">{title}</div>
          {subtitle ? <div className="text-slate-500 text-sm mt-1">{subtitle}</div> : null}
        </div>
        {right}
      </div>
      <div className="p-5">{children}</div>
    </div>
  );
}

function XaiDrawer({ open, onClose, skillEvidence, roleEvidence, selectedRole }) {
  return (
    <div className={classNames("fixed inset-0 z-50", open ? "" : "pointer-events-none")}>
      <div
        className={classNames(
          "absolute inset-0 bg-black/40 transition-opacity",
          open ? "opacity-100" : "opacity-0"
        )}
        onClick={onClose}
      />
      <div
        className={classNames(
          "absolute right-0 top-0 h-full w-full sm:w-[520px] bg-white shadow-xl border-l border-slate-200 transition-transform",
          open ? "translate-x-0" : "translate-x-full"
        )}
      >
        <div className="p-5 border-b border-slate-100 flex items-center justify-between">
          <div>
            <div className="text-slate-900 font-semibold">Explainability (XAI)</div>
            <div className="text-slate-500 text-sm">Evidence from transcript and validation</div>
          </div>
          <button
            className="px-3 py-1.5 rounded-lg border border-slate-200 text-slate-700 hover:bg-slate-50"
            onClick={onClose}
          >
            Close
          </button>
        </div>

        <div className="p-5 space-y-6 overflow-auto h-[calc(100%-72px)]">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <div className="font-semibold text-slate-900">Skill evidence</div>
              <Badge tone="blue">Transcript</Badge>
            </div>

            {!skillEvidence?.skills?.length ? (
              <div className="text-slate-600 text-sm">No skill evidence loaded yet.</div>
            ) : (
              <div className="space-y-3">
                {skillEvidence.skills.slice(0, 12).map((s) => (
                  <div key={s.skill} className="border border-slate-200 rounded-lg p-3">
                    <div className="flex items-center justify-between">
                      <div className="font-semibold text-slate-900">{s.skill}</div>
                      <Badge tone="slate">{s.level || "Unknown"}</Badge>
                    </div>
                    <div className="text-slate-500 text-sm mt-1">
                      Why: {s.reason || "Mapped from completed modules"}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div>
            <div className="flex items-center gap-2 mb-2">
              <div className="font-semibold text-slate-900">Role evidence</div>
              <Badge tone="green">Job data</Badge>
            </div>

            {!roleEvidence?.role ? (
              <div className="text-slate-600 text-sm">
                No role evidence loaded yet. Select a role in “Role suggestions” and click Explain.
              </div>
            ) : (
              <div className="border border-slate-200 rounded-lg p-3">
                <div className="flex items-center justify-between">
                  <div className="font-semibold text-slate-900">{roleEvidence.role}</div>
                  <Badge tone="amber">{selectedRole || "Selected role"}</Badge>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function normalizeSkillRow(s) {
  const skill = s.Skill ?? s.skill ?? "";
  const level = s.FinalSkillLevel ?? s.SkillLevel ?? s.level ?? "";
  const score = Number(s.FinalScore ?? s.ScoreNormalized ?? s.score ?? 0);
  const evidenceCount = s.EvidenceCount ?? s.evidence_count ?? s.Evidence ?? 0;

  return { skill, level, score, evidenceCount, raw: s };
}

export default function App() {
  const [studentId, setStudentId] = useState("IT21013928");
  const [regNo, setRegNo] = useState("");
  const [file, setFile] = useState(null);

  const [transcriptDetails, setTranscriptDetails] = useState(null);
  const [transcriptCourses, setTranscriptCourses] = useState([]);

  const [skillsResp, setSkillsResp] = useState(null);
  const [rolesResp, setRolesResp] = useState(null);

  const [quiz, setQuiz] = useState(null);
  const [quizResult, setQuizResult] = useState(null);

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  // XAI drawer
  const [xaiOpen, setXaiOpen] = useState(false);
  const [skillEvidence, setSkillEvidence] = useState(null);
  const [roleEvidence, setRoleEvidence] = useState(null);
  const [selectedRole, setSelectedRole] = useState("");

  // NEW: skill selection (max 5)
  const [selectedSkills, setSelectedSkills] = useState([]);
  const [numPerSkill, setNumPerSkill] = useState(3);
  const [difficulty, setDifficulty] = useState("mixed"); // easy|medium|hard|mixed

  const normalizedSkills = useMemo(() => {
    const arr = skillsResp?.skills || [];
    return arr.map(normalizeSkillRow);
  }, [skillsResp]);

  const topRoles = useMemo(() => rolesResp?.roles || [], [rolesResp]);

  async function safeRun(fn) {
    setError("");
    setBusy(true);
    try {
      await fn();
    } catch (e) {
      setError(e?.message || "Something went wrong");
    } finally {
      setBusy(false);
    }
  }

  function toggleSkill(skillName) {
    setError("");
    setSelectedSkills((prev) => {
      if (prev.includes(skillName)) return prev.filter((s) => s !== skillName);
      if (prev.length >= 5) {
        setError("You can select a maximum of 5 skills.");
        return prev;
      }
      return [...prev, skillName];
    });
  }

  function resetSelection() {
    setSelectedSkills([]);
  }

  const onUpload = async () => {
    if (!file) {
      setError("Please choose a PDF first.");
      return;
    }

    await safeRun(async () => {
      const out = await uploadTranscript(studentId.trim(), file, regNo);

      setTranscriptDetails(out.transcript_details || null);
      setTranscriptCourses(out.courses || []);

      const s = await getSkills(studentId.trim());
      setSkillsResp(s);

      try {
        const r = await getRoles(studentId.trim());
        setRolesResp(r);
      } catch {
        setRolesResp(null);
      }

      setQuiz(null);
      setQuizResult(null);

      // NEW: clear selection after a new upload
      setSelectedSkills([]);
    });
  };

  const onLoadSkills = async () => {
    await safeRun(async () => {
      const s = await getSkills(studentId.trim());
      setSkillsResp(s);
    });
  };

  const onLoadRoles = async () => {
    await safeRun(async () => setRolesResp(await getRoles(studentId.trim())));
  };

  // UPDATED: prepare quiz from selected skills
  const onPrepareQuiz = async () => {
    if (!selectedSkills.length) {
      setError("Select 1 to 5 skills first (from the Skills panel).");
      return;
    }

    await safeRun(async () => {
      const payload = {
        selected_skills: selectedSkills,
        num_questions_per_skill: Number(numPerSkill) || 3,
        difficulty,
      };

      const q = await prepareQuiz(studentId.trim(), payload);
      setQuiz(q);
      setQuizResult(null);
    });
  };

  const onSubmitQuiz = async () => {
    if (!quiz?.questions?.length) {
      setError("Prepare a quiz first.");
      return;
    }

    const responses = [];
    for (const q of quiz.questions) {
      const qid = q.QuestionID ?? q.question_id;
      const name = `q_${qid}`;
      const chosen = document.querySelector(`input[name="${name}"]:checked`);
      if (chosen) {
        responses.push({
          question_id: Number(qid),
          selected_option: chosen.value,
          response_time_seconds: 30,
        });
      }
    }

    if (!responses.length) {
      setError("Please answer at least one question before submitting.");
      return;
    }

    await safeRun(async () => {
      const result = await submitQuiz(studentId.trim(), responses);
      setQuizResult(result);

      setSkillsResp(await getSkills(studentId.trim()));
      try {
        setRolesResp(await getRoles(studentId.trim()));
      } catch {
        // roles may not exist for this student yet
      }
    });
  };

  const onOpenXai = async () => {
    await safeRun(async () => {
      const se = await getSkillEvidence(studentId.trim());
      setSkillEvidence(se);

      if (selectedRole) {
        const re = await getRoleEvidence(studentId.trim(), selectedRole);
        setRoleEvidence(re);
      } else {
        setRoleEvidence(null);
      }
      setXaiOpen(true);
    });
  };

  const onExplainRole = async (roleName) => {
    setSelectedRole(roleName);
    await safeRun(async () => {
      const re = await getRoleEvidence(studentId.trim(), roleName);
      setRoleEvidence(re);
      setXaiOpen(true);
    });
  };

  return (
    <div className="min-h-screen bg-slate-50">
      <div className="max-w-6xl mx-auto px-4 py-8">
        <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4 mb-6">
          <div>
            <div className="text-2xl font-bold text-slate-900">Transcript-Based Skill Validation</div>
            <div className="text-slate-600 mt-1">
              Upload transcript → View modules → Select skills → Quiz validation → Role recommendations
            </div>
          </div>

          <div className="flex gap-2">
            <button
              className="px-4 py-2 rounded-lg bg-slate-900 text-white hover:bg-slate-800 disabled:opacity-60"
              onClick={onOpenXai}
              disabled={busy}
              title="Show explainability panel"
            >
              Explain (XAI)
            </button>
            <a
              className="px-4 py-2 rounded-lg border border-slate-200 bg-white hover:bg-slate-50"
              href="http://127.0.0.1:8000/docs"
              target="_blank"
              rel="noreferrer"
            >
              API Docs
            </a>
          </div>
        </div>

        {error ? (
          <div className="mb-6 p-3 rounded-lg border border-red-200 bg-red-50 text-red-800 text-sm">
            {error}
          </div>
        ) : null}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1 space-y-6">
            <Card
              title="1) Upload transcript"
              subtitle="Upload PDF to extract details and build skill profile"
              right={busy ? <Badge tone="amber">Working</Badge> : <Badge tone="green">Ready</Badge>}
            >
              <div className="space-y-3">
                <div>
                  <label className="text-sm font-medium text-slate-700">Student ID</label>
                  <input
                    className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-200"
                    value={studentId}
                    onChange={(e) => setStudentId(e.target.value)}
                  />
                </div>

                <div>
                  <label className="text-sm font-medium text-slate-700">Reg No (optional)</label>
                  <input
                    className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-200"
                    value={regNo}
                    onChange={(e) => setRegNo(e.target.value)}
                    placeholder="Defaults to Student ID"
                  />
                </div>

                <div>
                  <label className="text-sm font-medium text-slate-700">Transcript file</label>
                  <input
                    type="file"
                    className="mt-1 w-full text-sm"
                    accept=".pdf,image/*"
                    onChange={(e) => setFile(e.target.files?.[0] || null)}
                  />
                </div>

                <button
                  className="w-full px-4 py-2 rounded-lg bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-60"
                  onClick={onUpload}
                  disabled={busy}
                >
                  Upload and Extract
                </button>
              </div>
            </Card>

            <Card title="2) Build view panels" subtitle="Load computed skills and roles">
              <div className="flex flex-wrap gap-2">
                <button
                  className="px-3 py-2 rounded-lg bg-white border border-slate-200 hover:bg-slate-50 disabled:opacity-60"
                  onClick={onLoadSkills}
                  disabled={busy}
                >
                  Load Skills
                </button>
                <button
                  className="px-3 py-2 rounded-lg bg-white border border-slate-200 hover:bg-slate-50 disabled:opacity-60"
                  onClick={onLoadRoles}
                  disabled={busy}
                >
                  Load Roles
                </button>

                {/* UPDATED: this now uses selected skills */}
                <button
                  className="px-3 py-2 rounded-lg bg-white border border-slate-200 hover:bg-slate-50 disabled:opacity-60"
                  onClick={onPrepareQuiz}
                  disabled={busy}
                  title="Generate quiz for selected skills"
                >
                  Generate Quiz
                </button>
              </div>

              <div className="mt-3 text-sm text-slate-500">
                Upload transcript first, then select up to 5 skills from the Skills panel, then click Generate Quiz.
              </div>

              {/* Quiz settings */}
              <div className="mt-4 grid grid-cols-1 gap-3">
                <div>
                  <label className="text-sm font-medium text-slate-700">Questions per skill</label>
                  <input
                    className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-200"
                    type="number"
                    min={1}
                    max={10}
                    value={numPerSkill}
                    onChange={(e) => setNumPerSkill(e.target.value)}
                  />
                </div>

                <div>
                  <label className="text-sm font-medium text-slate-700">Difficulty</label>
                  <select
                    className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-200"
                    value={difficulty}
                    onChange={(e) => setDifficulty(e.target.value)}
                  >
                    <option value="mixed">Mixed</option>
                    <option value="easy">Easy</option>
                    <option value="medium">Medium</option>
                    <option value="hard">Hard</option>
                  </select>
                </div>
              </div>
            </Card>

            <Card title="Selected skills" subtitle="Choose up to 5 skills to validate">
              {!selectedSkills.length ? (
                <div className="text-slate-600 text-sm">No skills selected yet.</div>
              ) : (
                <div className="flex flex-wrap gap-2">
                  {selectedSkills.map((s) => (
                    <span key={s} className="px-2 py-1 rounded-lg bg-slate-100 text-slate-700 text-sm">
                      {s}
                    </span>
                  ))}
                </div>
              )}

              <div className="mt-3 flex gap-2">
                <button
                  className="px-3 py-2 rounded-lg bg-white border border-slate-200 hover:bg-slate-50 disabled:opacity-60"
                  onClick={resetSelection}
                  disabled={busy || !selectedSkills.length}
                >
                  Clear
                </button>
              </div>
            </Card>
          </div>

          <div className="lg:col-span-2 space-y-6">
            <Card title="Transcript content" subtitle="Details and detected modules from the uploaded transcript">
              {!transcriptDetails && !transcriptCourses.length ? (
                <div className="text-slate-600 text-sm">Upload a transcript to see details and modules here.</div>
              ) : (
                <div className="space-y-4">
                  <div>
                    <div className="font-semibold text-slate-900">Transcript details</div>
                    <pre className="mt-2 text-xs bg-slate-900 text-slate-100 p-3 rounded-lg overflow-auto max-h-56">
                      {transcriptDetails ? JSON.stringify(transcriptDetails, null, 2) : "No details."}
                    </pre>
                  </div>

                  <div>
                    <div className="font-semibold text-slate-900">
                      Modules detected ({transcriptCourses.length})
                    </div>
                    {!transcriptCourses.length ? (
                      <div className="text-slate-600 text-sm mt-2">No modules detected.</div>
                    ) : (
                      <div className="mt-2 max-h-64 overflow-auto border border-slate-200 rounded-lg">
                        <table className="w-full text-sm">
                          <thead className="bg-slate-50">
                            <tr>
                              <th className="text-left p-2">Code</th>
                              <th className="text-left p-2">Name</th>
                              <th className="text-left p-2">Grade</th>
                            </tr>
                          </thead>
                          <tbody>
                            {transcriptCourses.map((c, idx) => (
                              <tr key={idx} className="border-t">
                                <td className="p-2">{c.CourseCode || c.code || "-"}</td>
                                <td className="p-2">{c.CourseName || c.name || "-"}</td>
                                <td className="p-2">{c.Grade || c.grade || "-"}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </Card>

            {/* UPDATED: Skills panel is now clickable/selectable */}
            <Card
              title="Skills inferred from transcript"
              subtitle="Select up to 5 skills you want to validate in the quiz"
              right={normalizedSkills.length ? <Badge tone="blue">{normalizedSkills.length} shown</Badge> : null}
            >
              {!normalizedSkills.length ? (
                <div className="text-slate-600 text-sm">No skills loaded yet.</div>
              ) : (
                <div className="space-y-3">
                  <div className="text-sm text-slate-600">
                    Selected: <span className="font-semibold text-slate-900">{selectedSkills.length}</span>/5
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {normalizedSkills.map((s) => {
                      const checked = selectedSkills.includes(s.skill);
                      return (
                        <button
                          key={s.skill}
                          type="button"
                          onClick={() => toggleSkill(s.skill)}
                          className={classNames(
                            "text-left border rounded-lg p-3 transition",
                            checked ? "border-blue-400 bg-blue-50" : "border-slate-200 hover:bg-slate-50"
                          )}
                          disabled={busy}
                          title="Click to select/unselect"
                        >
                          <div className="flex items-center justify-between gap-3">
                            <div className="flex items-center gap-2">
                              <input
                                type="checkbox"
                                readOnly
                                checked={checked}
                                className="pointer-events-none"
                              />
                              <div className="font-semibold text-slate-900">{s.skill}</div>
                            </div>
                            <Badge tone="slate">{s.level || "Unknown"}</Badge>
                          </div>
                          <div className="text-slate-500 text-sm mt-1">
                            Score: <span className="font-medium text-slate-700">{Number(s.score).toFixed(3)}</span>{" "}
                            Evidence: <span className="font-medium text-slate-700">{s.evidenceCount}</span>
                          </div>
                        </button>
                      );
                    })}
                  </div>

                  <div className="flex flex-wrap gap-2 pt-2">
                    <button
                      className="px-3 py-2 rounded-lg bg-slate-900 text-white hover:bg-slate-800 disabled:opacity-60"
                      onClick={onPrepareQuiz}
                      disabled={busy || !selectedSkills.length}
                      title="Generate quiz for selected skills"
                    >
                      Generate Quiz from Selected Skills
                    </button>
                    <button
                      className="px-3 py-2 rounded-lg bg-white border border-slate-200 hover:bg-slate-50 disabled:opacity-60"
                      onClick={resetSelection}
                      disabled={busy || !selectedSkills.length}
                    >
                      Clear selection
                    </button>
                  </div>
                </div>
              )}
            </Card>

            <Card
              title="Role suggestions"
              subtitle="Based on skills and job post templates"
              right={topRoles.length ? <Badge tone="green">{topRoles.length} shown</Badge> : null}
            >
              {!topRoles.length ? (
                <div className="text-slate-600 text-sm">
                  No roles loaded yet. (Roles require running the role model pipeline for this student.)
                </div>
              ) : (
                <div className="space-y-3">
                  {topRoles.slice(0, 10).map((r, idx) => (
                    <div key={idx} className="border border-slate-200 rounded-lg p-3">
                      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                        <div>
                          <div className="font-semibold text-slate-900">{r.role_name || r.Role || r.role}</div>
                        </div>
                        <button
                          className="px-3 py-2 rounded-lg bg-white border border-slate-200 hover:bg-slate-50"
                          onClick={() => onExplainRole(r.role_name || r.Role || r.role)}
                        >
                          Explain
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </Card>

            <Card title="Quiz validation" subtitle="Validate claimed skills (based on what you selected)">
              {!quiz?.questions?.length ? (
                <div className="text-slate-600 text-sm">
                  Select skills and click “Generate Quiz” to build a quiz.
                </div>
              ) : (
                <div className="space-y-5">
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge tone="blue">Selected skills</Badge>
                    {(quiz.selected_skills || selectedSkills).slice(0, 5).map((s) => (
                      <span key={s} className="px-2 py-0.5 rounded-full text-xs bg-slate-100 text-slate-700">
                        {s}
                      </span>
                    ))}
                  </div>

                  {quiz.questions.map((q, idx) => {
                    const qid = q.QuestionID ?? q.question_id;
                    const name = `q_${qid}`;
                    return (
                      <div key={qid} className="border border-slate-200 rounded-lg p-4">
                        <div className="font-semibold text-slate-900">
                          Q{idx + 1}. {q.QuestionText}
                        </div>

                        <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-2">
                          {["A", "B", "C", "D"].map((opt) => {
                            const text = q[`Option${opt}`];
                            if (!text) return null;
                            return (
                              <label
                                key={opt}
                                className="flex items-start gap-2 p-2 rounded-lg border border-slate-200 hover:bg-slate-50 cursor-pointer"
                              >
                                <input type="radio" name={name} value={opt} className="mt-1" />
                                <div className="text-sm text-slate-800">
                                  <span className="font-semibold mr-2">{opt}.</span>
                                  {text}
                                </div>
                              </label>
                            );
                          })}
                        </div>
                      </div>
                    );
                  })}

                  <button
                    className="px-4 py-2 rounded-lg bg-slate-900 text-white hover:bg-slate-800 disabled:opacity-60"
                    onClick={onSubmitQuiz}
                    disabled={busy}
                  >
                    Submit Quiz
                  </button>

                  {quizResult ? (
                    <div className="mt-3 border border-slate-200 rounded-lg p-4 bg-slate-50">
                      <div className="font-semibold text-slate-900">Result</div>
                      <pre className="mt-2 text-xs overflow-auto">{JSON.stringify(quizResult, null, 2)}</pre>
                    </div>
                  ) : null}
                </div>
              )}
            </Card>
          </div>
        </div>
      </div>

      <XaiDrawer
        open={xaiOpen}
        onClose={() => setXaiOpen(false)}
        skillEvidence={skillEvidence}
        roleEvidence={roleEvidence}
        selectedRole={selectedRole}
      />
    </div>
  );
}
