import { useMemo } from "react";

function GradeBadge({ grade }) {
  const getGradeColor = (grade) => {
    if (!grade) return "bg-slate-100 text-slate-700";
    const g = grade.toUpperCase();
    if (g.startsWith("A")) return "bg-green-100 text-green-800 border-green-300";
    if (g.startsWith("B")) return "bg-blue-100 text-blue-800 border-blue-300";
    if (g.startsWith("C")) return "bg-yellow-100 text-yellow-800 border-yellow-300";
    if (g.startsWith("D")) return "bg-orange-100 text-orange-800 border-orange-300";
    return "bg-red-100 text-red-800 border-red-300";
  };

  return (
    <span
      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold border ${getGradeColor(
        grade
      )}`}
    >
      {grade || "N/A"}
    </span>
  );
}

function TranscriptDisplay({ details, courses, skills, studentId, numCourses }) {
  // Group courses by year if Year column exists
  const coursesByYear = useMemo(() => {
    if (!courses?.length) return {};
    
    const grouped = {};
    courses.forEach((course) => {
      const year = course.Year || course.year || "Other";
      if (!grouped[year]) {
        grouped[year] = [];
      }
      grouped[year].push(course);
    });
    
    // Sort years
    const sortedYears = Object.keys(grouped).sort((a, b) => {
      if (a === "Other") return 1;
      if (b === "Other") return -1;
      return a.localeCompare(b);
    });
    
    const result = {};
    sortedYears.forEach((year) => {
      result[year] = grouped[year];
    });
    
    return result;
  }, [courses]);

  // Calculate statistics
  const stats = useMemo(() => {
    if (!courses?.length) {
      return { totalCourses: 0, avgGrade: null, gradeDistribution: {} };
    }

    const gradePoints = {
      A: 4.0,
      "A-": 3.7,
      "B+": 3.3,
      B: 3.0,
      "B-": 2.7,
      "C+": 2.3,
      C: 2.0,
      "C-": 1.7,
      "D+": 1.3,
      D: 1.0,
      "D-": 0.7,
      F: 0.0,
    };

    let totalPoints = 0;
    let count = 0;
    const distribution = {};

    courses.forEach((course) => {
      const grade = (course.Grade || course.grade || "").trim().toUpperCase();
      if (grade) {
        // Count distribution
        const baseGrade = grade.replace(/[+-]/, "");
        distribution[baseGrade] = (distribution[baseGrade] || 0) + 1;

        // Calculate points
        const points = gradePoints[grade] || gradePoints[baseGrade] || 0;
        if (points > 0) {
          totalPoints += points;
          count++;
        }
      }
    });

    return {
      totalCourses: courses.length,
      avgGrade: count > 0 ? (totalPoints / count).toFixed(2) : null,
      gradeDistribution: distribution,
    };
  }, [courses]);

  const hasYearGrouping = Object.keys(coursesByYear).length > 1;

  return (
    <div className="space-y-6">
      {/* Student Information Card */}
      {details && (
        <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl p-6 border border-blue-200 shadow-sm">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <h3 className="text-2xl font-bold text-slate-900 mb-2">
                {details.candidate_name || details.name || "Student Name"}
              </h3>
              <div className="space-y-1 text-slate-600">
                {details.programme && (
                  <div className="flex items-center gap-2">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                    </svg>
                    <span className="font-medium">{details.programme}</span>
                  </div>
                )}
                {studentId && (
                  <div className="flex items-center gap-2">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V8a2 2 0 00-2-2h-5m-4 0V5a2 2 0 114 0v1m-4 0a2 2 0 104 0m-5 8a2 2 0 100-4 2 2 0 000 4zm0 0c1.306 0 2.417.835 2.83 2M9 14a3.001 3.001 0 00-2.83 2M15 11h3m-3 4h2" />
                    </svg>
                    <span>ID: {studentId}</span>
                  </div>
                )}
              </div>
            </div>
            <div className="text-right">
              <div className="text-sm text-slate-500 mb-1">Total Courses</div>
              <div className="text-3xl font-bold text-blue-600">{stats.totalCourses || numCourses || 0}</div>
            </div>
          </div>
        </div>
      )}

      {/* Academic Statistics */}
      {stats.totalCourses > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {stats.avgGrade && (
            <div className="bg-white rounded-lg p-4 border border-slate-200 shadow-sm">
              <div className="text-sm text-slate-500 mb-1">Average Grade Point</div>
              <div className="text-2xl font-bold text-slate-900">{stats.avgGrade}</div>
            </div>
          )}
          <div className="bg-white rounded-lg p-4 border border-slate-200 shadow-sm">
            <div className="text-sm text-slate-500 mb-1">Courses Completed</div>
            <div className="text-2xl font-bold text-slate-900">{stats.totalCourses}</div>
          </div>
          {skills?.length > 0 && (
            <div className="bg-white rounded-lg p-4 border border-slate-200 shadow-sm">
              <div className="text-sm text-slate-500 mb-1">Skills Identified</div>
              <div className="text-2xl font-bold text-slate-900">{skills.length}</div>
            </div>
          )}
        </div>
      )}

      {/* Grade Distribution */}
      {Object.keys(stats.gradeDistribution).length > 0 && (
        <div className="bg-white rounded-lg p-4 border border-slate-200 shadow-sm">
          <h4 className="font-semibold text-slate-900 mb-3">Grade Distribution</h4>
          <div className="flex flex-wrap gap-2">
            {Object.entries(stats.gradeDistribution)
              .sort(([a], [b]) => {
                const order = { A: 1, B: 2, C: 3, D: 4, F: 5 };
                return (order[a] || 99) - (order[b] || 99);
              })
              .map(([grade, count]) => (
                <div key={grade} className="flex items-center gap-2 bg-slate-50 rounded-lg px-3 py-2">
                  <span className="font-semibold text-slate-700">{grade}</span>
                  <span className="text-slate-500 text-sm">{count}</span>
                </div>
              ))}
          </div>
        </div>
      )}

      {/* Courses Table */}
      {courses?.length > 0 && (
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
          <div className="px-6 py-4 border-b border-slate-200 bg-slate-50">
            <h3 className="text-lg font-semibold text-slate-900">Academic Transcript</h3>
            <p className="text-sm text-slate-500 mt-1">
              {stats.totalCourses} courses detected and processed
            </p>
          </div>

          <div className="overflow-x-auto">
            {hasYearGrouping ? (
              // Grouped by year
              <div className="divide-y divide-slate-200">
                {Object.entries(coursesByYear).map(([year, yearCourses]) => (
                  <div key={year} className="p-4">
                    <div className="mb-3 flex items-center gap-2">
                      <h4 className="font-semibold text-slate-900">
                        {year === "Other" ? "Other Courses" : `Year ${year}`}
                      </h4>
                      <span className="text-xs text-slate-500 bg-slate-100 px-2 py-1 rounded">
                        {yearCourses.length} {yearCourses.length === 1 ? "course" : "courses"}
                      </span>
                    </div>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b border-slate-200">
                            <th className="text-left py-2 px-3 font-semibold text-slate-700">Code</th>
                            <th className="text-left py-2 px-3 font-semibold text-slate-700">Course Title</th>
                            <th className="text-center py-2 px-3 font-semibold text-slate-700">Grade</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100">
                          {yearCourses.map((course, idx) => (
                            <tr key={idx} className="hover:bg-slate-50 transition-colors">
                              <td className="py-3 px-3 font-mono text-slate-900 font-medium">
                                {course.CourseCode || course.code || "-"}
                              </td>
                              <td className="py-3 px-3 text-slate-700">
                                {course.CourseTitle || course.title || course.CourseName || course.name || "-"}
                              </td>
                              <td className="py-3 px-3 text-center">
                                <GradeBadge grade={course.Grade || course.grade} />
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              // Simple table without year grouping
              <table className="w-full text-sm">
                <thead className="bg-slate-50">
                  <tr>
                    <th className="text-left py-3 px-4 font-semibold text-slate-700">Course Code</th>
                    <th className="text-left py-3 px-4 font-semibold text-slate-700">Course Title</th>
                    <th className="text-center py-3 px-4 font-semibold text-slate-700">Grade</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {courses.map((course, idx) => (
                    <tr key={idx} className="hover:bg-slate-50 transition-colors">
                      <td className="py-3 px-4 font-mono text-slate-900 font-medium">
                        {course.CourseCode || course.code || "-"}
                      </td>
                      <td className="py-3 px-4 text-slate-700">
                        {course.CourseTitle || course.title || course.CourseName || course.name || "-"}
                      </td>
                      <td className="py-3 px-4 text-center">
                        <GradeBadge grade={course.Grade || course.grade} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      )}

      {/* Skills Preview */}
      {skills?.length > 0 && (
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
          <div className="px-6 py-4 border-b border-slate-200 bg-slate-50">
            <h3 className="text-lg font-semibold text-slate-900">Extracted Skills Preview</h3>
            <p className="text-sm text-slate-500 mt-1">
              Top skills identified from your transcript
            </p>
          </div>
          <div className="p-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {skills.slice(0, 12).map((skill, idx) => {
                const score = Number(skill.ScoreNormalized ?? skill.FinalScore ?? 0);
                const scorePercent = (score * 100).toFixed(1);
                return (
                  <div
                    key={idx}
                    className="border border-slate-200 rounded-lg p-4 hover:border-blue-300 hover:shadow-md transition-all"
                  >
                    <div className="flex items-start justify-between mb-2">
                      <h4 className="font-semibold text-slate-900 text-sm line-clamp-2">
                        {skill.Skill || skill.skill || "Skill"}
                      </h4>
                    </div>
                    <div className="flex items-center gap-2 mt-2">
                      <div className="flex-1 bg-slate-200 rounded-full h-2 overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-blue-500 to-indigo-600 transition-all"
                          style={{ width: `${Math.min(scorePercent, 100)}%` }}
                        />
                      </div>
                      <span className="text-xs font-medium text-slate-600 min-w-[3rem] text-right">
                        {scorePercent}%
                      </span>
                    </div>
                    {skill.FinalSkillLevel && (
                      <div className="mt-2">
                        <span className="text-xs px-2 py-1 rounded bg-blue-50 text-blue-700 font-medium">
                          {skill.FinalSkillLevel}
                        </span>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
            {skills.length > 12 && (
              <div className="mt-4 text-center text-sm text-slate-500">
                And {skills.length - 12} more skills...
              </div>
            )}
          </div>
        </div>
      )}

      {!courses?.length && !details && (
        <div className="text-center py-12 text-slate-500">
          <svg
            className="mx-auto h-12 w-12 text-slate-400 mb-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            />
          </svg>
          <p>No transcript data to display</p>
        </div>
      )}
    </div>
  );
}

export default TranscriptDisplay;

