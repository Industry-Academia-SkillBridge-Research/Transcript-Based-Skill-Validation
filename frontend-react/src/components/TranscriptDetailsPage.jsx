import { useMemo } from "react";
import TranscriptDisplay from "./TranscriptDisplay";

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

export default function TranscriptDetailsPage({ details, courses, studentId, onContinue, onBack, onViewSkills }) {
  // Group courses by year
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

  const hasYearGrouping = Object.keys(coursesByYear).length > 1;

  const studentName = details?.candidate_name || details?.name || "N/A";
  const programme = details?.programme || details?.program || "N/A";
  const specialization = details?.specialization || details?.field_of_specialization || "";

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      {/* Header with Back button */}
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
        <div className="flex gap-3">
          {onViewSkills && (
            <button
              onClick={onViewSkills}
              className="px-6 py-2.5 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-lg font-semibold hover:from-indigo-700 hover:to-purple-700 transition-all shadow-lg hover:shadow-xl"
            >
              View Skills
            </button>
          )}
          <button
            onClick={onContinue}
            className="px-6 py-2.5 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-lg font-semibold hover:from-blue-700 hover:to-indigo-700 transition-all shadow-lg hover:shadow-xl"
          >
            Continue to Dashboard
          </button>
        </div>
      </div>

      {/* Student Information Card */}
      <div className="bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 rounded-2xl p-8 border border-blue-200 shadow-lg">
        <div className="text-center mb-6">
          <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-gradient-to-br from-blue-500 to-indigo-600 mb-4 shadow-lg">
            <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
            </svg>
          </div>
          <h1 className="text-3xl font-bold text-slate-900 mb-2">{studentName}</h1>
          <div className="flex items-center justify-center space-x-4 text-slate-600">
            {studentId && (
              <div className="flex items-center space-x-2">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V8a2 2 0 00-2-2h-5m-4 0V5a2 2 0 114 0v1m-4 0a2 2 0 104 0m-5 8a2 2 0 100-4 2 2 0 000 4zm0 0c1.306 0 2.417.835 2.83 2M9 14a3.001 3.001 0 00-2.83 2M15 11h3m-3 4h2" />
                </svg>
                <span className="font-medium">ID: {studentId}</span>
              </div>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
          <div className="bg-white/80 backdrop-blur-sm rounded-xl p-4 border border-blue-100">
            <div className="text-xs text-slate-500 uppercase tracking-wide mb-1">Degree Programme</div>
            <div className="text-lg font-semibold text-slate-900">{programme}</div>
          </div>
          {specialization && (
            <div className="bg-white/80 backdrop-blur-sm rounded-xl p-4 border border-blue-100">
              <div className="text-xs text-slate-500 uppercase tracking-wide mb-1">Specialization</div>
              <div className="text-lg font-semibold text-slate-900">{specialization}</div>
            </div>
          )}
        </div>
      </div>

      {/* Academic Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm">
          <div className="text-sm text-slate-500 mb-2">Total Courses</div>
          <div className="text-3xl font-bold text-blue-600">{courses?.length || 0}</div>
        </div>
        <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm">
          <div className="text-sm text-slate-500 mb-2">Modules Completed</div>
          <div className="text-3xl font-bold text-indigo-600">{courses?.length || 0}</div>
        </div>
        <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm">
          <div className="text-sm text-slate-500 mb-2">Transcript Status</div>
          <div className="text-lg font-semibold text-green-600">✓ Processed</div>
        </div>
      </div>

      {/* Modules and Grades Section */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-lg overflow-hidden">
        <div className="px-8 py-6 border-b border-slate-200 bg-gradient-to-r from-slate-50 to-blue-50">
          <h2 className="text-2xl font-bold text-slate-900">Academic Transcript</h2>
          <p className="text-slate-600 mt-1">
            {courses?.length || 0} modules detected and processed
          </p>
        </div>

        <div className="p-6">
          {hasYearGrouping ? (
            // Grouped by year
            <div className="space-y-6">
              {Object.entries(coursesByYear).map(([year, yearCourses]) => (
                <div key={year} className="border-b border-slate-100 last:border-b-0 pb-6 last:pb-0">
                  <div className="mb-4 flex items-center gap-3">
                    <h3 className="text-xl font-bold text-slate-900">
                      {year === "Other" ? "Other Courses" : `Year ${year}`}
                    </h3>
                    <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm font-medium">
                      {yearCourses.length} {yearCourses.length === 1 ? "course" : "courses"}
                    </span>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b-2 border-slate-200">
                          <th className="text-left py-3 px-4 font-bold text-slate-700">Course Code</th>
                          <th className="text-left py-3 px-4 font-bold text-slate-700">Course Title</th>
                          <th className="text-center py-3 px-4 font-bold text-slate-700">Grade</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {yearCourses.map((course, idx) => (
                          <tr key={idx} className="hover:bg-slate-50 transition-colors">
                            <td className="py-4 px-4 font-mono text-slate-900 font-semibold">
                              {course.CourseCode || course.code || "-"}
                            </td>
                            <td className="py-4 px-4 text-slate-700">
                              {course.CourseTitle || course.title || course.CourseName || course.name || "-"}
                            </td>
                            <td className="py-4 px-4 text-center">
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
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b-2 border-slate-200">
                    <th className="text-left py-3 px-4 font-bold text-slate-700">Course Code</th>
                    <th className="text-left py-3 px-4 font-bold text-slate-700">Course Title</th>
                    <th className="text-center py-3 px-4 font-bold text-slate-700">Grade</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {courses?.map((course, idx) => (
                    <tr key={idx} className="hover:bg-slate-50 transition-colors">
                      <td className="py-4 px-4 font-mono text-slate-900 font-semibold">
                        {course.CourseCode || course.code || "-"}
                      </td>
                      <td className="py-4 px-4 text-slate-700">
                        {course.CourseTitle || course.title || course.CourseName || course.name || "-"}
                      </td>
                      <td className="py-4 px-4 text-center">
                        <GradeBadge grade={course.Grade || course.grade} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

