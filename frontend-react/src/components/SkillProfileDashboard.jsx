import { useEffect, useState } from "react";
import { getSkills } from "../api";

function SkillProfileDashboard({ studentId, selectedSkills = [], onBack, onViewJobs }) {
  const [skillsData, setSkillsData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function loadSkillProfile() {
      try {
        setLoading(true);
        const response = await getSkills(studentId);
        setSkillsData(response);
        setError(null);
      } catch (err) {
        setError(err.message || "Failed to load skill profile");
      } finally {
        setLoading(false);
      }
    }

    if (studentId) {
      loadSkillProfile();
    }
  }, [studentId]);

  // Filter skills to only show quizzed skills if selectedSkills provided
  const quizzedSkills = skillsData?.skills
    ? selectedSkills.length > 0
      ? skillsData.skills.filter((s) =>
          selectedSkills.some(
            (selected) => selected.toLowerCase() === (s.Skill || s.skill || "").toLowerCase()
          )
        )
      : skillsData.skills
    : [];

  const getLevelBadgeColor = (level) => {
    const levelStr = String(level || "").toLowerCase();
    if (levelStr.includes("advanced")) return "bg-purple-100 text-purple-800 border-purple-300";
    if (levelStr.includes("proficient")) return "bg-blue-100 text-blue-800 border-blue-300";
    if (levelStr.includes("developing")) return "bg-yellow-100 text-yellow-800 border-yellow-300";
    if (levelStr.includes("beginner")) return "bg-green-100 text-green-800 border-green-300";
    return "bg-slate-100 text-slate-800 border-slate-300";
  };

  const formatScore = (score) => {
    if (score === null || score === undefined || isNaN(score)) return "N/A";
    return (Number(score) * 100).toFixed(1) + "%";
  };

  const getScoreColor = (score) => {
    if (score === null || score === undefined || isNaN(score)) return "text-slate-500";
    const numScore = Number(score);
    if (numScore >= 0.75) return "text-green-600";
    if (numScore >= 0.50) return "text-blue-600";
    if (numScore >= 0.25) return "text-yellow-600";
    return "text-red-600";
  };

  if (loading) {
    return (
      <div className="max-w-6xl mx-auto space-y-6 min-h-screen pb-12">
        <div className="text-center py-12">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-purple-600"></div>
          <p className="mt-4 text-slate-600">Loading skill profile...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-6xl mx-auto space-y-6 min-h-screen pb-12">
        <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-red-800">
          {error}
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6 min-h-screen pb-12">
      {/* Header */}
      <div className="flex items-center justify-between">
        <button
          onClick={onBack}
          className="flex items-center space-x-2 px-4 py-2 text-slate-600 hover:text-slate-900 hover:bg-white rounded-lg transition-colors"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          <span>Back</span>
        </button>
        {onViewJobs && (
          <button
            onClick={onViewJobs}
            className="flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-lg font-semibold hover:from-indigo-700 hover:to-purple-700 transition-all shadow-lg hover:shadow-xl"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M21 13.255A23.931 23.931 0 0112 15c-3.183 0-6.22-.62-9-1.745M16 6V4a2 2 0 00-2-2h-4a2 2 0 00-2 2v2m4 6h.01M5 20h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
              />
            </svg>
            <span>View Job Recommendations</span>
          </button>
        )}
      </div>

      {/* Title Section */}
      <div className="text-center mb-8">
        <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 mb-4 shadow-lg">
          <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
            />
          </svg>
        </div>
        <h1 className="text-4xl font-bold text-slate-900 mb-2">Skill Profile Dashboard</h1>
        <p className="text-slate-600 text-lg">Verified skills after quiz validation</p>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <div className="bg-white rounded-xl p-6 border-2 border-blue-200 shadow-md text-center">
          <div className="text-sm text-slate-600 mb-2 font-medium">Total Skills</div>
          <div className="text-4xl font-bold text-blue-600">{quizzedSkills.length}</div>
          <div className="text-xs text-slate-500">validated</div>
        </div>
        <div className="bg-white rounded-xl p-6 border-2 border-green-200 shadow-md text-center">
          <div className="text-sm text-slate-600 mb-2 font-medium">Avg Transcript</div>
          <div className="text-4xl font-bold text-green-600">
            {quizzedSkills.length > 0
              ? formatScore(
                  quizzedSkills.reduce(
                    (sum, s) =>
                      sum +
                      (s.ScoreNormalized !== null && s.ScoreNormalized !== undefined
                        ? Number(s.ScoreNormalized)
                        : 0),
                    0
                  ) / quizzedSkills.length
                )
              : "0%"}
          </div>
          <div className="text-xs text-slate-500">score</div>
        </div>
        <div className="bg-white rounded-xl p-6 border-2 border-purple-200 shadow-md text-center">
          <div className="text-sm text-slate-600 mb-2 font-medium">Avg Quiz</div>
          <div className="text-4xl font-bold text-purple-600">
            {quizzedSkills.filter((s) => s.QuizProficiency !== null && s.QuizProficiency !== undefined).length > 0
              ? formatScore(
                  quizzedSkills
                    .filter((s) => s.QuizProficiency !== null && s.QuizProficiency !== undefined)
                    .reduce((sum, s) => sum + Number(s.QuizProficiency), 0) /
                    quizzedSkills.filter((s) => s.QuizProficiency !== null && s.QuizProficiency !== undefined).length
                )
              : "N/A"}
          </div>
          <div className="text-xs text-slate-500">score</div>
        </div>
        <div className="bg-white rounded-xl p-6 border-2 border-indigo-200 shadow-md text-center">
          <div className="text-sm text-slate-600 mb-2 font-medium">Avg Final</div>
          <div className="text-4xl font-bold text-indigo-600">
            {quizzedSkills.filter((s) => s.FinalScore !== null && s.FinalScore !== undefined).length > 0
              ? formatScore(
                  quizzedSkills
                    .filter((s) => s.FinalScore !== null && s.FinalScore !== undefined)
                    .reduce((sum, s) => sum + Number(s.FinalScore), 0) /
                    quizzedSkills.filter((s) => s.FinalScore !== null && s.FinalScore !== undefined).length
                )
              : "N/A"}
          </div>
          <div className="text-xs text-slate-500">score</div>
        </div>
      </div>

      {/* Skills Table */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-lg overflow-hidden">
        <div className="px-6 py-5 border-b border-slate-200 bg-gradient-to-r from-slate-50 to-indigo-50/30">
          <h3 className="text-xl font-bold text-slate-900">Skill Breakdown</h3>
          <p className="text-sm text-slate-600 mt-1">Transcript scores, quiz scores, and final verified levels</p>
        </div>

        {quizzedSkills.length === 0 ? (
          <div className="p-8 text-center text-slate-500">
            No skills found. Please complete a quiz first.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-slate-50 border-b border-slate-200">
                <tr>
                  <th className="px-6 py-4 text-left text-xs font-semibold text-slate-700 uppercase tracking-wider">
                    Skill
                  </th>
                  <th className="px-6 py-4 text-center text-xs font-semibold text-slate-700 uppercase tracking-wider">
                    Level
                  </th>
                  <th className="px-6 py-4 text-center text-xs font-semibold text-slate-700 uppercase tracking-wider">
                    Transcript Score
                  </th>
                  <th className="px-6 py-4 text-center text-xs font-semibold text-slate-700 uppercase tracking-wider">
                    Quiz Score
                  </th>
                  <th className="px-6 py-4 text-center text-xs font-semibold text-slate-700 uppercase tracking-wider">
                    Final Score
                  </th>
                  <th className="px-6 py-4 text-center text-xs font-semibold text-slate-700 uppercase tracking-wider">
                    Questions
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200">
                {quizzedSkills.map((skill, idx) => {
                  const transcriptScore = skill.ScoreNormalized ?? skill.score_normalized ?? null;
                  const quizScore = skill.QuizProficiency ?? skill.quiz_proficiency ?? null;
                  const finalScore = skill.FinalScore ?? skill.final_score ?? null;
                  const skillLevel =
                    skill.FinalSkillLevel ?? skill.final_skill_level ?? skill.SkillLevel ?? skill.skill_level ?? "Unknown";
                  const numQuestions = skill.NumQuestions ?? skill.num_questions ?? 0;
                  const skillName = skill.Skill ?? skill.skill ?? "Unknown";

                  return (
                    <tr key={idx} className="hover:bg-slate-50 transition-colors">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm font-semibold text-slate-900">{skillName}</div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-center">
                        <span
                          className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold border ${getLevelBadgeColor(
                            skillLevel
                          )}`}
                        >
                          {skillLevel}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-center">
                        <div className={`text-lg font-bold ${getScoreColor(transcriptScore)}`}>
                          {formatScore(transcriptScore)}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-center">
                        {quizScore !== null && quizScore !== undefined ? (
                          <div className={`text-lg font-bold ${getScoreColor(quizScore)}`}>
                            {formatScore(quizScore)}
                          </div>
                        ) : (
                          <div className="text-sm text-slate-400">Not quizzed</div>
                        )}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-center">
                        <div className={`text-lg font-bold ${getScoreColor(finalScore)}`}>
                          {formatScore(finalScore)}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-center">
                        <div className="text-sm text-slate-600">{numQuestions > 0 ? numQuestions : "-"}</div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h4 className="font-semibold text-blue-900 mb-2">Understanding the Scores</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-blue-800">
          <div>
            <strong>Transcript Score:</strong> Skill level inferred from your academic transcript (courses and grades).
          </div>
          <div>
            <strong>Quiz Score:</strong> Your performance on the validation quiz for this skill.
          </div>
          <div>
            <strong>Final Score:</strong> Combined score using dynamic weighting (transcript + quiz validation).
          </div>
        </div>
      </div>
    </div>
  );
}

export default SkillProfileDashboard;
