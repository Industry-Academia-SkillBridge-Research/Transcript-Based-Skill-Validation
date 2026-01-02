import React, { useMemo, useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import {
  getSkills,
  getRoles,
  prepareQuiz,
  submitQuiz,
  uploadTranscript,
  uploadTranscriptAuto,
  getSkillEvidence,
  getRoleEvidence,
} from "./api";
import FileUpload from "./components/FileUpload";
import TranscriptDetailsPage from "./components/TranscriptDetailsPage";
import SkillsPage from "./components/SkillsPage";
import QuizPage from "./components/QuizPage";
import QuizResultPage from "./components/QuizResultPage";
import SkillProfileDashboard from "./components/SkillProfileDashboard";
import JobRecommendations from "./components/JobRecommendations";

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

function Card({ title, subtitle, right, children, className = "" }) {
  return (
    <div className={`bg-white/90 backdrop-blur-sm rounded-2xl border border-slate-200/50 shadow-xl hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-1 ${className}`}>
      {title && (
        <div className="px-6 py-5 border-b border-slate-200/50 bg-gradient-to-r from-slate-50 to-blue-50/30 rounded-t-2xl flex items-center justify-between">
          <div>
            <h3 className="text-xl font-bold text-slate-900 mb-1">{title}</h3>
            {subtitle && <p className="text-sm text-slate-600">{subtitle}</p>}
          </div>
          {right && <div>{right}</div>}
        </div>
      )}
      <div className="p-6">{children}</div>
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
  const navigate = useNavigate();
  const location = useLocation();
  const [studentId, setStudentId] = useState("IT21013928");
  const [regNo, setRegNo] = useState("");
  const [file, setFile] = useState(null);
  const [transcriptUploaded, setTranscriptUploaded] = useState(false);
  const [showTranscriptDetails, setShowTranscriptDetails] = useState(false);
  const [showSkillsPage, setShowSkillsPage] = useState(false);
  const [showQuizPage, setShowQuizPage] = useState(false);
  const [showQuizResultPage, setShowQuizResultPage] = useState(false);
  const [showSkillDashboard, setShowSkillDashboard] = useState(false);
  const [showJobRecommendations, setShowJobRecommendations] = useState(false);
  const [selectedQuizSkills, setSelectedQuizSkills] = useState([]);
  const [quizQuestions, setQuizQuestions] = useState(null);

  const [transcriptDetails, setTranscriptDetails] = useState(null);
  const [transcriptCourses, setTranscriptCourses] = useState([]);
  const [uploadedSkills, setUploadedSkills] = useState([]);

  const [skillsResp, setSkillsResp] = useState(null);
  const [rolesResp, setRolesResp] = useState(null);

  const [quiz, setQuiz] = useState(null);
  const [quizResult, setQuizResult] = useState(null);
  const [currentQuizId, setCurrentQuizId] = useState(null);
  const [quizTimeLimit, setQuizTimeLimit] = useState(null);
  const [quizSessionToken, setQuizSessionToken] = useState(null);
  const [quizSessionStartTime, setQuizSessionStartTime] = useState(null);

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
      // Use auto-upload endpoint that extracts student ID from transcript
      const out = await uploadTranscriptAuto(file, null);
      
      // Extract student ID from response and update state
      const extractedStudentId = out.student_id || out.transcript_details?.student_id || "";
      if (extractedStudentId) {
        setStudentId(extractedStudentId);
      }

      setTranscriptDetails(out.transcript_details || null);
      setTranscriptCourses(out.courses || []);
      setUploadedSkills(out.skills || []);

      const finalStudentId = extractedStudentId || studentId.trim();
      if (finalStudentId) {
        const s = await getSkills(finalStudentId);
        setSkillsResp(s);

        try {
          const r = await getRoles(finalStudentId);
          setRolesResp(r);
        } catch {
          setRolesResp(null);
        }
      }

      setQuiz(null);
      setQuizResult(null);

      // NEW: clear selection after a new upload
      setSelectedSkills([]);
      
      // Show transcript details page first
      setShowTranscriptDetails(true);
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
      
      // Store quiz data
      setCurrentQuizId(q.quiz_id);
      setQuizTimeLimit(q.time_limit_minutes);
      setQuizSessionToken(q.session_token);
      setQuizSessionStartTime(q.session_start_time);
      setQuiz({ ...q, quiz_id: q.quiz_id });
      setQuizResult(null);
      
      // Navigate to quiz page
      setSelectedQuizSkills(selectedSkills);
      setShowSkillsPage(false);
      setShowQuizPage(true);
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
      // Include quiz_id if available
      const result = await submitQuiz(
        studentId.trim(),
        responses,
        currentQuizId || quiz?.quiz_id || null
      );
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
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 relative overflow-hidden">
      {/* Decorative background elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-purple-300 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse-slow"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-blue-300 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse-slow" style={{ animationDelay: '2s' }}></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-80 h-80 bg-indigo-300 rounded-full mix-blend-multiply filter blur-xl opacity-10 animate-float"></div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 relative z-10">
        {transcriptUploaded && (
          <div className="mb-8">
            <div className="bg-white/80 backdrop-blur-md rounded-2xl p-6 shadow-xl border border-white/20 mb-6">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                <div className="flex-1">
                  <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-600 bg-clip-text text-transparent mb-2">
                    Transcript-Based Skill Validation
                  </h1>
                  <div className="flex flex-wrap items-center gap-2 text-slate-600">
                    <span className="flex items-center gap-1">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                      </svg>
                      Upload transcript
                    </span>
                    <span className="text-slate-400">→</span>
                    <span className="flex items-center gap-1">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                      </svg>
                      View modules
                    </span>
                    <span className="text-slate-400">→</span>
                    <span className="flex items-center gap-1">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z" />
                      </svg>
                      Select skills
                    </span>
                    <span className="text-slate-400">→</span>
                    <span className="flex items-center gap-1">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                      Quiz validation
                    </span>
                    <span className="text-slate-400">→</span>
                    <span className="flex items-center gap-1">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 13.255A23.931 23.931 0 0112 15c-3.183 0-6.22-.62-9-1.745M16 6V4a2 2 0 00-2-2h-4a2 2 0 00-2 2v2m4 6h.01M5 20h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                      </svg>
                      Role recommendations
                    </span>
                  </div>
                </div>

                <div className="flex gap-3">
                  <button
                    className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-purple-600 to-indigo-600 text-white font-semibold hover:from-purple-700 hover:to-indigo-700 disabled:opacity-60 transition-all shadow-lg hover:shadow-xl transform hover:-translate-y-0.5"
                    onClick={onOpenXai}
                    disabled={busy}
                    title="Show explainability panel"
                  >
                    <span className="flex items-center gap-2">
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                      </svg>
                      Explain (XAI)
                    </span>
                  </button>
                  <a
                    className="px-5 py-2.5 rounded-xl border-2 border-slate-300 bg-white/90 backdrop-blur-sm text-slate-700 font-semibold hover:bg-white hover:border-slate-400 transition-all shadow-md hover:shadow-lg transform hover:-translate-y-0.5"
                    href="http://127.0.0.1:8000/docs"
                    target="_blank"
                    rel="noreferrer"
                  >
                    <span className="flex items-center gap-2">
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                      </svg>
                      API Docs
                    </span>
                  </a>
                </div>
              </div>
            </div>
          </div>
        )}

        {error ? (
          <div className="mb-6 p-4 rounded-xl border-2 border-red-300 bg-gradient-to-r from-red-50 to-pink-50 text-red-800 text-sm shadow-lg animate-in fade-in slide-in-from-top-2">
            <div className="flex items-center gap-3">
              <svg className="w-5 h-5 text-red-600 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span className="font-medium">{error}</span>
            </div>
          </div>
        ) : null}

        {/* Show job recommendations */}
        {showJobRecommendations ? (
          <JobRecommendations
            studentId={studentId}
            onBack={() => {
              setShowJobRecommendations(false);
              setShowSkillDashboard(true);
            }}
          />
        ) : showSkillDashboard ? (
          <SkillProfileDashboard
            studentId={studentId}
            selectedSkills={selectedQuizSkills}
            onBack={() => {
              setShowSkillDashboard(false);
              setShowQuizResultPage(true);
            }}
            onViewJobs={() => {
              setShowSkillDashboard(false);
              setShowJobRecommendations(true);
            }}
          />
        ) : showQuizResultPage && quizResult && quizQuestions ? (
          <QuizResultPage
            studentId={studentId}
            quizResult={quizResult}
            questions={quizQuestions}
            selectedSkills={selectedQuizSkills}
            onBack={() => {
              setShowQuizResultPage(false);
              setShowQuizPage(false);
              setShowSkillsPage(true);
            }}
            onViewDashboard={() => {
              setShowQuizResultPage(false);
              setShowSkillDashboard(true);
            }}
            onRetakeQuiz={() => {
              setQuizResult(null);
              setQuizQuestions(null);
              setShowQuizResultPage(false);
              setShowQuizPage(true);
            }}
          />
        ) : showQuizPage && selectedQuizSkills.length > 0 ? (
          <QuizPage
            studentId={studentId}
            selectedSkills={selectedQuizSkills}
            quizId={currentQuizId || quiz?.quiz_id}
            timeLimitMinutes={quizTimeLimit}
            sessionToken={quizSessionToken}
            sessionStartTime={quizSessionStartTime}
            onBack={() => {
              setShowQuizPage(false);
              setShowSkillsPage(true);
            }}
            onQuizCompleted={(result, questions) => {
              setQuizResult(result);
              setQuizQuestions(questions);
              setShowQuizPage(false);
              setShowQuizResultPage(true);
            }}
          />
        ) : showSkillsPage && uploadedSkills.length > 0 ? (
          /* Show skills page */
          <SkillsPage
            skills={uploadedSkills}
            studentName={transcriptDetails?.candidate_name || transcriptDetails?.name}
            studentId={studentId}
            onContinue={() => {
              setShowSkillsPage(false);
              setTranscriptUploaded(true);
            }}
            onGenerateQuiz={(skills) => {
              setSelectedQuizSkills(skills);
              setShowSkillsPage(false);
              setShowQuizPage(true);
            }}
            onBack={() => {
              setShowSkillsPage(false);
              setShowTranscriptDetails(true);
            }}
          />
        ) : showTranscriptDetails && transcriptDetails ? (
          /* Show transcript details page after upload */
          <TranscriptDetailsPage
            details={transcriptDetails}
            courses={transcriptCourses}
            studentId={studentId}
            onContinue={() => {
              setShowTranscriptDetails(false);
              setTranscriptUploaded(true);
            }}
            onViewSkills={() => {
              setShowTranscriptDetails(false);
              setShowSkillsPage(true);
            }}
            onBack={() => {
              setShowTranscriptDetails(false);
              setFile(null);
              setTranscriptDetails(null);
              setTranscriptCourses([]);
              setUploadedSkills([]);
            }}
          />
        ) : !transcriptUploaded ? (
          /* Show only upload form if transcript not uploaded yet */
          <div className="max-w-3xl mx-auto">
            <div className="text-center mb-8">
              <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent mb-2">
                Transcript-Based Skill Validation
              </h1>
              <p className="text-slate-600 text-lg">
                Upload your academic transcript to analyze your skills and discover career opportunities
              </p>
            </div>
            <Card
              title=""
              subtitle=""
              right={busy ? <Badge tone="amber">Processing...</Badge> : <Badge tone="green">Ready</Badge>}
            >
              <div className="space-y-6">
                <FileUpload
                  onFileSelect={setFile}
                  loading={busy}
                  accept=".pdf,image/*"
                />

                <button
                  className="w-full px-6 py-3.5 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-xl font-semibold hover:from-blue-700 hover:to-indigo-700 disabled:opacity-60 disabled:cursor-not-allowed transition-all shadow-lg hover:shadow-xl transform hover:scale-[1.02] text-lg"
                  onClick={onUpload}
                  disabled={busy || !file}
                >
                  {busy ? (
                    <span className="flex items-center justify-center space-x-2">
                      <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      <span>Uploading and Processing...</span>
                    </span>
                  ) : (
                    "Upload and Extract"
                  )}
                </button>
                
                <div className="flex items-center justify-center space-x-2 text-sm text-slate-500 pt-2">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span>Student ID will be automatically extracted from your transcript</span>
                </div>
              </div>
            </Card>
          </div>
        ) : (
          /* Show all sections after transcript is uploaded */
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-1 space-y-6">
              <Card
                title="1) Upload transcript"
                subtitle="Upload PDF to extract details and build skill profile"
                right={busy ? <Badge tone="amber">Working</Badge> : <Badge tone="green">Ready</Badge>}
              >
                <div className="space-y-3">
                  {/* Student ID and Reg No fields hidden - will be extracted from transcript automatically */}
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
                  {topRoles.slice(0, 10).map((r, idx) => {
                    const roleName = r.role_name || r.Role || r.role;
                    const readiness = (r.readiness_score || r.ReadinessScore || 0) * 100;
                    const coverage = (r.coverage || r.Coverage || 0) * 100;
                    const numSkills = r.num_skills || r.NumSkills || 0;
                    const numPresent = r.num_skills_present || r.NumSkillsPresent || 0;
                    const numWeakMissing = r.num_weak_or_missing || r.NumWeakOrMissing || 0;
                    const weakMissingSkills = r.weak_or_missing_skills || r.WeakOrMissingSkills || "";
                    const weakMissingList = weakMissingSkills ? weakMissingSkills.split(", ").filter(s => s.trim()) : [];

                    return (
                      <div key={idx} className="border border-slate-200 rounded-lg p-4 hover:border-blue-300 hover:shadow-md transition-all">
                        <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3">
                          <div className="flex-1">
                            <div className="flex items-center gap-3 mb-2">
                              <div className="font-semibold text-slate-900 text-lg">{roleName}</div>
                              <Badge tone={readiness >= 70 ? "green" : readiness >= 50 ? "blue" : "amber"}>
                                {readiness.toFixed(1)}% Ready
                              </Badge>
                            </div>
                            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm mb-3">
                              <div>
                                <div className="text-slate-500">Coverage</div>
                                <div className="font-semibold text-slate-900">{coverage.toFixed(1)}%</div>
                              </div>
                              <div>
                                <div className="text-slate-500">Skills Match</div>
                                <div className="font-semibold text-slate-900">{numPresent}/{numSkills}</div>
                              </div>
                              <div>
                                <div className="text-slate-500">Gaps</div>
                                <div className="font-semibold text-red-600">{numWeakMissing}</div>
                              </div>
                              <div>
                                <div className="text-slate-500">Missing</div>
                                <div className="font-semibold text-orange-600">{numWeakMissing > 0 ? "Yes" : "No"}</div>
                              </div>
                            </div>
                            {weakMissingList.length > 0 && (
                              <div className="mt-2 pt-3 border-t border-slate-100">
                                <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
                                  Missing or Weak Skills:
                                </div>
                                <div className="flex flex-wrap gap-2">
                                  {weakMissingList.slice(0, 5).map((skill, skillIdx) => (
                                    <span key={skillIdx} className="text-xs px-2 py-1 bg-red-50 text-red-700 rounded border border-red-200">
                                      {skill}
                                    </span>
                                  ))}
                                  {weakMissingList.length > 5 && (
                                    <span className="text-xs px-2 py-1 bg-slate-100 text-slate-600 rounded">
                                      +{weakMissingList.length - 5} more
                                    </span>
                                  )}
                                </div>
                              </div>
                            )}
                          </div>
                          <button
                            className="px-4 py-2 rounded-lg bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-semibold hover:from-blue-700 hover:to-indigo-700 transition-all shadow-sm hover:shadow-md whitespace-nowrap"
                            onClick={() => onExplainRole(roleName)}
                          >
                            View Details
                          </button>
                        </div>
                      </div>
                    );
                  })}
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
        )}
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
