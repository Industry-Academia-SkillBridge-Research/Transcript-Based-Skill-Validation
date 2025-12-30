import { useState } from "react";
import { uploadTranscript } from "../api";

export default function UploadTranscript({ studentId, regNo, onChangeStudentId, onChangeRegNo, onUploaded }) {
  const [file, setFile] = useState(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [details, setDetails] = useState(null);
  const [courses, setCourses] = useState([]);
  const [skills, setSkills] = useState([]);

  async function onSubmit(e) {
    e.preventDefault();
    setError("");
    setDetails(null);
    setCourses([]);
    setSkills([]);

    if (!String(studentId || "").trim()) {
      setError("Please enter Student ID.");
      return;
    }
    if (!file) {
      setError("Please choose a transcript PDF.");
      return;
    }

    try {
      setLoading(true);
      const data = await uploadTranscript(String(studentId).trim(), file, regNo || "");

      setDetails(data.transcript_details || data.details || null);
      setCourses(data.courses || []);
      setSkills(data.skills || []);

      onUploaded?.(data);
    } catch (err) {
      setError(err.message || "Upload failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200">
      <div className="px-5 py-4 border-b border-slate-100">
        <div className="text-slate-900 font-semibold">1) Upload transcript</div>
        <div className="text-slate-500 text-sm mt-1">Upload PDF to extract details and modules</div>
      </div>

      <div className="p-5">
        <form onSubmit={onSubmit} className="space-y-3">
          <div>
            <label className="text-sm font-medium text-slate-700">Student ID</label>
            <input
              className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-200"
              value={studentId}
              onChange={(e) => onChangeStudentId?.(e.target.value)}
              placeholder="IT210xxxxx"
            />
          </div>

          <div>
            <label className="text-sm font-medium text-slate-700">Reg No (optional)</label>
            <input
              className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-200"
              value={regNo}
              onChange={(e) => onChangeRegNo?.(e.target.value)}
              placeholder="Defaults to Student ID"
            />
          </div>

          <div>
            <label className="text-sm font-medium text-slate-700">Transcript file</label>
            <input
              type="file"
              className="mt-1 w-full text-sm"
              accept=".pdf"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full px-4 py-2 rounded-lg bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-60"
          >
            {loading ? "Uploading..." : "Upload and Extract"}
          </button>
        </form>

        {error ? (
          <div className="mt-3 p-3 rounded-lg border border-red-200 bg-red-50 text-red-800 text-sm whitespace-pre-wrap">
            {error}
          </div>
        ) : null}

        <div className="mt-5 space-y-4">
          <div>
            <div className="font-semibold text-slate-900">Transcript details</div>
            <pre className="mt-2 text-xs bg-slate-900 text-slate-100 p-3 rounded-lg overflow-auto max-h-56">
              {details ? JSON.stringify(details, null, 2) : "No data yet."}
            </pre>
          </div>

          <div>
            <div className="font-semibold text-slate-900">Detected modules</div>
            {!courses?.length ? (
              <div className="text-slate-600 text-sm mt-2">No modules yet.</div>
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
                    {courses.map((c, idx) => (
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

          <div>
            <div className="font-semibold text-slate-900">Claimed skills from upload response</div>
            {!skills?.length ? (
              <div className="text-slate-600 text-sm mt-2">No skills yet.</div>
            ) : (
              <div className="mt-2 grid grid-cols-1 md:grid-cols-2 gap-2">
                {skills.slice(0, 12).map((s, idx) => (
                  <div key={idx} className="border border-slate-200 rounded-lg p-3">
                    <div className="font-semibold text-slate-900">{s.Skill || s.skill || "Skill"}</div>
                    <div className="text-slate-600 text-sm mt-1">
                      Score: {Number(s.ScoreNormalized ?? s.FinalScore ?? 0).toFixed(3)} • Evidence:{" "}
                      {s.EvidenceCount ?? "-"}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
