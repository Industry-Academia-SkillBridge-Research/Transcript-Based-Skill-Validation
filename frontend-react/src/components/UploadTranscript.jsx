// src/components/UploadTranscript.jsx
import { useState } from "react";
import { uploadTranscript } from "../api";

export function UploadTranscript({ currentStudentId, onPreview }) {
  const [studentId, setStudentId] = useState(currentStudentId || "");
  const [regno, setRegno] = useState("");
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("No transcript uploaded yet.");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!studentId) {
      alert("Enter student ID for this transcript.");
      return;
    }
    if (!file) {
      alert("Choose a PDF/image transcript first.");
      return;
    }
    try {
      setLoading(true);
      setStatus("Uploading and processing transcript...");
      const result = await uploadTranscript(studentId, regno, file);
      setStatus(
        `Parsed ${result.skill_profile_rows} skill rows for student ${result.student_id}.`
      );
      if (onPreview && result.skills_preview) {
        onPreview(result); // send preview up to parent
      }
    } catch (err) {
      console.error(err);
      setStatus(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <h2>Upload transcript & build skill profile (from PDF/image)</h2>
      <p className="muted">
        This calls <code>/students/{{"{"}}student_id{{"}"}}/upload-transcript</code> and runs the
        transcript parsing + skill mapping pipeline on the uploaded file.
      </p>

      <form onSubmit={handleSubmit}>
        <div className="field-row">
          <label>Student ID for this transcript</label>
          <input
            type="text"
            value={studentId}
            onChange={(e) => setStudentId(e.target.value)}
          />
        </div>
        <div className="field-row">
          <label>Reg No (optional, will default to Student ID)</label>
          <input
            type="text"
            value={regno}
            onChange={(e) => setRegno(e.target.value)}
            placeholder="IT21xxxxx"
          />
        </div>
        <div className="field-row">
          <label>Transcript file (PDF / image)</label>
          <input
            type="file"
            accept=".pdf,image/*"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
          />
        </div>
        <button className="primary" type="submit" disabled={loading}>
          {loading ? "Processing..." : "Upload & Process Transcript"}
        </button>
      </form>

      <pre className="status-box">{status}</pre>
    </div>
  );
}
