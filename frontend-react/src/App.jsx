import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import { getSkills, getRoles } from './api';
import { UploadTranscript } from './components/UploadTranscript.jsx';
import { SkillTable } from './components/SkillTable.jsx';
import { QuizSection } from './components/QuizSection.jsx';
import { RoleTable } from './components/RoleTable.jsx';

function App() {
  const [studentId, setStudentId] = useState("IT21001288");

  const [mainSkills, setMainSkills] = useState(null);
  const [mainSkillsError, setMainSkillsError] = useState("");

  const [roles, setRoles] = useState(null);
  const [rolesError, setRolesError] = useState("");

  const [uploadPreview, setUploadPreview] = useState(null);

  async function handleLoadSkills() {
    const id = studentId.trim();
    if (!id) {
      alert("Enter a student ID first.");
      return;
    }
    try {
      setMainSkillsError("");
      const data = await getSkills(id);
      setMainSkills(data);
    } catch (err) {
      console.error(err);
      setMainSkills(null);
      setMainSkillsError(err.message);
    }
  }

  async function handleLoadRoles() {
    const id = studentId.trim();
    if (!id) {
      alert("Enter a student ID first.");
      return;
    }
    try {
      setRolesError("");
      const data = await getRoles(id);
      setRoles(data);
    } catch (err) {
      console.error(err);
      setRoles(null);
      setRolesError(err.message);
    }
  }

  return (
    <div className="app-root">
      <h1>Transcript-based Skill Validation – Demo UI (React)</h1>

      <div className="card">
        <div className="field-row">
          <label>Student ID (for main dataset & quiz)</label>
          <input
            type="text"
            value={studentId}
            onChange={(e) => setStudentId(e.target.value)}
          />
        </div>
        <div className="btn-row">
          <button className="primary" onClick={handleLoadSkills}>
            Load Skills (from dataset)
          </button>
          <button className="secondary" onClick={handleLoadRoles}>
            Load Role Matches
          </button>
        </div>
      </div>

      <UploadTranscript
        currentStudentId={studentId}
        onPreview={setUploadPreview}
      />

      <SkillTable
        title="Skill profile (from main dataset / fused pipeline)"
        data={mainSkills}
        error={mainSkillsError}
      />

      {uploadPreview && (
        <div className="card">
          <h2>Skill profile (just from uploaded transcript)</h2>
          <p className="muted">
            This is built only from the PDF/image you uploaded, using the
            same course → skill mapping.
          </p>
          <ul>
            {uploadPreview.skills_preview.map((s) => (
              <li key={s.skill}>
                <strong>{s.skill}</strong> – score {s.score.toFixed(3)} (
                {s.level})
              </li>
            ))}
          </ul>
        </div>
      )}

      <RoleTable
        title="Role matches (using fused skills + quiz)"
        data={roles}
        error={rolesError}
      />

      <QuizSection
        studentId={studentId}
        onAfterQuiz={async () => {
          // refresh main skills & roles after quiz
          try {
            await handleLoadSkills();
            await handleLoadRoles();
          } catch (_) {
            // ignore
          }
        }}
      />
    </div>
  );
}

export default App;