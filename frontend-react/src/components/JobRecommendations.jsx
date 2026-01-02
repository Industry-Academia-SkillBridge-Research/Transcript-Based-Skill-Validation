import { useEffect, useState } from "react";
import { getRoles, getRoleEvidence } from "../api";

function JobRecommendations({ studentId, onBack }) {
  const [rolesData, setRolesData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedRole, setSelectedRole] = useState(null);
  const [roleEvidence, setRoleEvidence] = useState(null);
  const [loadingEvidence, setLoadingEvidence] = useState(false);

  useEffect(() => {
    async function loadJobRoles() {
      try {
        setLoading(true);
        setError(null);
        const response = await getRoles(studentId);
        setRolesData(response);
      } catch (err) {
        setError(err.message || "Failed to load job recommendations. Make sure job roles are processed.");
      } finally {
        setLoading(false);
      }
    }

    if (studentId) {
      loadJobRoles();
    }
  }, [studentId]);

  const loadRoleDetails = async (roleName) => {
    if (selectedRole === roleName && roleEvidence) {
      setSelectedRole(null);
      setRoleEvidence(null);
      return;
    }

    try {
      setLoadingEvidence(true);
      const evidence = await getRoleEvidence(studentId, roleName);
      setRoleEvidence(evidence);
      setSelectedRole(roleName);
    } catch (err) {
      console.error("Failed to load role evidence:", err);
      setRoleEvidence(null);
    } finally {
      setLoadingEvidence(false);
    }
  };

  const getReadinessColor = (score) => {
    if (!score && score !== 0) return "text-slate-500";
    const numScore = Number(score);
    if (numScore >= 0.8) return "text-green-600";
    if (numScore >= 0.6) return "text-blue-600";
    if (numScore >= 0.4) return "text-yellow-600";
    return "text-red-600";
  };

  const getReadinessBadgeColor = (score) => {
    if (!score && score !== 0) return "bg-slate-100 text-slate-800 border-slate-300";
    const numScore = Number(score);
    if (numScore >= 0.8) return "bg-green-100 text-green-800 border-green-300";
    if (numScore >= 0.6) return "bg-blue-100 text-blue-800 border-blue-300";
    if (numScore >= 0.4) return "bg-yellow-100 text-yellow-800 border-yellow-300";
    return "bg-red-100 text-red-800 border-red-300";
  };

  const formatScore = (score) => {
    if (score === null || score === undefined || isNaN(score)) return "N/A";
    return (Number(score) * 100).toFixed(1) + "%";
  };

  const getReadinessLabel = (score) => {
    if (!score && score !== 0) return "Unknown";
    const numScore = Number(score);
    if (numScore >= 0.8) return "Excellent Match";
    if (numScore >= 0.6) return "Good Match";
    if (numScore >= 0.4) return "Moderate Match";
    return "Low Match";
  };

  if (loading) {
    return (
      <div className="max-w-6xl mx-auto space-y-6 min-h-screen pb-12">
        <div className="text-center py-12">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
          <p className="mt-4 text-slate-600">Loading job recommendations...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-6xl mx-auto space-y-6 min-h-screen pb-12">
        <div className="bg-red-50 border border-red-200 rounded-xl p-6 text-red-800">
          <div className="flex items-start gap-3">
            <svg className="w-6 h-6 text-red-600 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <div>
              <h3 className="font-semibold text-lg mb-2">Failed to Load Job Recommendations</h3>
              <p>{error}</p>
              <p className="mt-2 text-sm text-red-700">
                Make sure job roles have been processed from Job_data.json. You may need to run the job processing pipeline.
              </p>
            </div>
          </div>
        </div>
        {onBack && (
          <button
            onClick={onBack}
            className="flex items-center space-x-2 px-6 py-3 bg-slate-600 text-white rounded-lg font-semibold hover:bg-slate-700 transition-all"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            <span>Back</span>
          </button>
        )}
      </div>
    );
  }

  const roles = rolesData?.roles || [];

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
      </div>

      {/* Title Section */}
      <div className="text-center mb-8">
        <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 mb-4 shadow-lg">
          <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M21 13.255A23.931 23.931 0 0112 15c-3.183 0-6.22-.62-9-1.745M16 6V4a2 2 0 00-2-2h-4a2 2 0 00-2 2v2m4 6h.01M5 20h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
            />
          </svg>
        </div>
        <h1 className="text-4xl font-bold text-slate-900 mb-2">Job Recommendations</h1>
        <p className="text-slate-600 text-lg">Roles matched to your verified skill profile</p>
      </div>

      {/* Summary Stats */}
      {roles.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <div className="bg-white rounded-xl p-6 border-2 border-indigo-200 shadow-md text-center">
            <div className="text-sm text-slate-600 mb-2 font-medium">Total Roles</div>
            <div className="text-4xl font-bold text-indigo-600">{roles.length}</div>
            <div className="text-xs text-slate-500">matched</div>
          </div>
          <div className="bg-white rounded-xl p-6 border-2 border-green-200 shadow-md text-center">
            <div className="text-sm text-slate-600 mb-2 font-medium">Top Match</div>
            <div className="text-2xl font-bold text-green-600">
              {roles.length > 0 ? formatScore(roles[0].ReadinessScore ?? roles[0].RoleReadiness ?? roles[0].Score) : "N/A"}
            </div>
            <div className="text-xs text-slate-500">
              {roles.length > 0 ? roles[0].RoleName ?? "Unknown" : ""}
            </div>
          </div>
          <div className="bg-white rounded-xl p-6 border-2 border-blue-200 shadow-md text-center">
            <div className="text-sm text-slate-600 mb-2 font-medium">Avg Readiness</div>
            <div className="text-4xl font-bold text-blue-600">
              {roles.length > 0
                ? formatScore(
                    roles.reduce(
                      (sum, r) => sum + (Number(r.ReadinessScore ?? r.RoleReadiness ?? r.Score ?? 0) || 0),
                      0
                    ) / roles.length
                  )
                : "0%"}
            </div>
            <div className="text-xs text-slate-500">across all roles</div>
          </div>
        </div>
      )}

      {/* Roles List */}
      {roles.length === 0 ? (
        <div className="bg-white rounded-xl border border-slate-200 shadow-lg p-8 text-center">
          <svg className="w-16 h-16 text-slate-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <h3 className="text-xl font-semibold text-slate-900 mb-2">No Job Roles Found</h3>
          <p className="text-slate-600 mb-4">
            Job roles need to be processed from Job_data.json. Please run the job processing pipeline first.
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {roles.map((role, idx) => {
            const readinessScore = role.ReadinessScore ?? role.RoleReadiness ?? role.Score ?? 0;
            const roleName = role.RoleName ?? "Unknown Role";
            const coverage = role.Coverage ?? role.SkillCoverage ?? null;
            const missingSkills = role.MissingSkills || role.WeakSkills || [];
            const missingSkillsList = typeof missingSkills === "string" 
              ? missingSkills.split(",").map(s => s.trim()).filter(s => s)
              : Array.isArray(missingSkills) 
                ? missingSkills 
                : [];

            return (
              <div
                key={idx}
                className="bg-white rounded-xl border border-slate-200 shadow-lg overflow-hidden hover:shadow-xl transition-shadow"
              >
                <div className="p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex-1">
                      <h3 className="text-2xl font-bold text-slate-900 mb-2">{roleName}</h3>
                      <div className="flex items-center gap-4 flex-wrap">
                        <div className="flex items-center gap-2">
                          <span className="text-sm text-slate-600">Readiness:</span>
                          <span className={`text-lg font-bold ${getReadinessColor(readinessScore)}`}>
                            {formatScore(readinessScore)}
                          </span>
                          <span
                            className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold border ${getReadinessBadgeColor(
                              readinessScore
                            )}`}
                          >
                            {getReadinessLabel(readinessScore)}
                          </span>
                        </div>
                        {coverage !== null && (
                          <div className="flex items-center gap-2">
                            <span className="text-sm text-slate-600">Skill Coverage:</span>
                            <span className="text-lg font-semibold text-slate-700">{formatScore(coverage)}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Missing/Weak Skills */}
                  {missingSkillsList.length > 0 && (
                    <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-4">
                      <div className="flex items-start gap-2 mb-2">
                        <svg className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                        </svg>
                        <div className="flex-1">
                          <h4 className="font-semibold text-yellow-900 mb-1">Skills to Improve</h4>
                          <div className="flex flex-wrap gap-2 mt-2">
                            {missingSkillsList.map((skill, skillIdx) => (
                              <span
                                key={skillIdx}
                                className="px-2 py-1 bg-yellow-100 text-yellow-800 rounded text-xs font-medium"
                              >
                                {skill}
                              </span>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* View Details Button */}
                  <button
                    onClick={() => loadRoleDetails(roleName)}
                    disabled={loadingEvidence}
                    className="flex items-center gap-2 px-4 py-2 bg-indigo-100 text-indigo-700 rounded-lg font-semibold hover:bg-indigo-200 transition-all disabled:opacity-50"
                  >
                    {loadingEvidence && selectedRole === roleName ? (
                      <>
                        <div className="inline-block animate-spin rounded-full h-4 w-4 border-b-2 border-indigo-700"></div>
                        <span>Loading...</span>
                      </>
                    ) : selectedRole === roleName ? (
                      <>
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
                        </svg>
                        <span>Hide Details</span>
                      </>
                    ) : (
                      <>
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                        </svg>
                        <span>View Skill Details</span>
                      </>
                    )}
                  </button>

                  {/* Role Evidence/Details */}
                  {selectedRole === roleName && roleEvidence && (
                    <div className="mt-4 border-t border-slate-200 pt-4">
                      <h4 className="font-semibold text-slate-900 mb-3">Skill Breakdown</h4>
                      {roleEvidence.required_skills && roleEvidence.required_skills.length > 0 ? (
                        <div className="space-y-2">
                          {roleEvidence.required_skills.map((gap, gapIdx) => (
                            <div
                              key={gapIdx}
                              className={`p-3 rounded-lg border ${
                                gap.is_weak_or_missing
                                  ? "bg-red-50 border-red-200"
                                  : "bg-green-50 border-green-200"
                              }`}
                            >
                              <div className="flex items-center justify-between flex-wrap gap-2">
                                <span className="font-medium text-slate-900">{gap.skill}</span>
                                <div className="flex items-center gap-3 flex-wrap">
                                  <span className="text-sm text-slate-600">
                                    Required: <strong>{formatScore(gap.required_importance)}</strong>
                                  </span>
                                  <span className="text-sm text-slate-600">
                                    Your Score: <strong>{formatScore(gap.student_score)}</strong>
                                  </span>
                                  {gap.student_level && (
                                    <span className="text-xs text-slate-500">Level: {gap.student_level}</span>
                                  )}
                                  {gap.is_weak_or_missing && (
                                    <span className="px-2 py-1 bg-red-100 text-red-800 rounded text-xs font-medium">
                                      Needs Improvement
                                    </span>
                                  )}
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p className="text-slate-600 text-sm">No detailed skill breakdown available.</p>
                      )}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default JobRecommendations;
