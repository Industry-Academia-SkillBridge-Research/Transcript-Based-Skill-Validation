import { useState } from "react";
import { uploadTranscriptAuto } from "../api";
import TranscriptDisplay from "./TranscriptDisplay";

export default function UploadTranscript({ onUploaded }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [uploaded, setUploaded] = useState(false);

  const [details, setDetails] = useState(null);
  const [courses, setCourses] = useState([]);
  const [skills, setSkills] = useState([]);
  const [numCourses, setNumCourses] = useState(0);
  const [extractedStudentId, setExtractedStudentId] = useState(null);

  async function onSubmit(e) {
    e.preventDefault();
    setError("");

    if (!file) {
      setError("Please choose a transcript PDF.");
      return;
    }

    try {
      setLoading(true);
      
      // Use the auto-upload endpoint that extracts student ID from transcript
      const data = await uploadTranscriptAuto(file, null);

      // Extract student ID from response
      const finalStudentId = data.student_id || data.transcript_details?.student_id || "";
      setExtractedStudentId(finalStudentId);

      setDetails(data.transcript_details || data.details || null);
      setCourses(data.courses || []);
      setSkills(data.skills || []);
      setNumCourses(data.num_courses_detected || data.courses?.length || 0);
      setUploaded(true);

      onUploaded?.(data);
    } catch (err) {
      setError(err.message || "Upload failed");
    } finally {
      setLoading(false);
    }
  }

  function handleReset() {
    setFile(null);
    setDetails(null);
    setCourses([]);
    setSkills([]);
    setExtractedStudentId(null);
    setUploaded(false);
    setError("");
  }

  // If transcript is uploaded, only show the display
  if (uploaded && (details || courses?.length > 0 || skills?.length > 0)) {
    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between bg-white rounded-xl shadow-sm border border-slate-200 px-5 py-4">
          <div>
            <div className="text-slate-900 font-semibold">Transcript Uploaded Successfully</div>
            <div className="text-slate-500 text-sm mt-1">
              {extractedStudentId && `Student ID: ${extractedStudentId}`}
            </div>
          </div>
          <button
            onClick={handleReset}
            className="px-4 py-2 rounded-lg border border-slate-200 text-slate-700 hover:bg-slate-50 text-sm font-medium"
          >
            Upload Another
          </button>
        </div>
        <TranscriptDisplay
          details={details}
          courses={courses}
          skills={skills}
          studentId={extractedStudentId}
          numCourses={numCourses}
        />
      </div>
    );
  }

  // Show upload form
  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200">
      <div className="px-5 py-4 border-b border-slate-100">
        <div className="text-slate-900 font-semibold">Upload Transcript</div>
        <div className="text-slate-500 text-sm mt-1">Upload PDF to extract details and modules</div>
      </div>

      <div className="p-5">
        <form onSubmit={onSubmit} className="space-y-4">
          <div>
            <label className="text-sm font-medium text-slate-700 block mb-2">
              Transcript file (PDF)
            </label>
            <input
              type="file"
              className="w-full text-sm border border-slate-200 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-200"
              accept=".pdf"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
            {file && (
              <div className="mt-2 text-sm text-slate-600">
                Selected: <span className="font-medium">{file.name}</span>
              </div>
            )}
          </div>

          <button
            type="submit"
            disabled={loading || !file}
            className="w-full px-4 py-2 rounded-lg bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-60 disabled:cursor-not-allowed font-medium"
          >
            {loading ? "Uploading and Processing..." : "Upload and Extract"}
          </button>
        </form>

        {error && (
          <div className="mt-3 p-3 rounded-lg border border-red-200 bg-red-50 text-red-800 text-sm whitespace-pre-wrap">
            {error}
          </div>
        )}
      </div>
    </div>
  );
}
