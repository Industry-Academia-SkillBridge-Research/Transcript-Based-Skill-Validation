// src/components/SkillTable.jsx
import { useMemo, useState } from "react";

export function SkillTable({ title, data, error, onGenerateQuiz }) {
  const skills = useMemo(() => {
    if (!data || !data.skills) return [];
    return data.skills.map((s) => ({
      skill: s.skill,
      score: s.score,
      evidence_count: s.evidence_count,
      level: s.level,
    }));
  }, [data]);

  const [selectedSkills, setSelectedSkills] = useState([]);
  const [numPerSkill, setNumPerSkill] = useState(3);
  const [difficulty, setDifficulty] = useState("mixed");
  const [localMsg, setLocalMsg] = useState("");

  function toggleSkill(skillName) {
    setLocalMsg("");
    setSelectedSkills((prev) => {
      if (prev.includes(skillName)) {
        return prev.filter((s) => s !== skillName);
      }
      if (prev.length >= 5) {
        setLocalMsg("You can select a maximum of 5 skills.");
        return prev;
      }
      return [...prev, skillName];
    });
  }

  function handleGenerate() {
    setLocalMsg("");
    if (!onGenerateQuiz) {
      setLocalMsg("Quiz generator handler not connected.");
      return;
    }
    if (selectedSkills.length === 0) {
      setLocalMsg("Select at least 1 skill to generate a quiz.");
      return;
    }

    onGenerateQuiz({
      selected_skills: selectedSkills,
      num_questions_per_skill: Number(numPerSkill) || 3,
      difficulty,
    });
  }

  return (
    <div className="card">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12 }}>
        <h2 style={{ margin: 0 }}>{title}</h2>
        <div className="muted" style={{ fontSize: 13 }}>
          Selected: {selectedSkills.length}/5
        </div>
      </div>

      {error && <div className="error-box">{error}</div>}
      {localMsg && <div className="error-box">{localMsg}</div>}

      {!skills || skills.length === 0 ? (
        <div className="muted">No skills found.</div>
      ) : (
        <>
          <table>
            <thead>
              <tr>
                <th style={{ width: 70 }}>Pick</th>
                <th>Skill</th>
                <th>Score</th>
                <th>Evidence</th>
                <th>Level</th>
              </tr>
            </thead>
            <tbody>
              {skills.map((s) => (
                <tr key={s.skill}>
                  <td>
                    <input
                      type="checkbox"
                      checked={selectedSkills.includes(s.skill)}
                      onChange={() => toggleSkill(s.skill)}
                    />
                  </td>
                  <td>{s.skill}</td>
                  <td>{Number(s.score).toFixed(3)}</td>
                  <td>{s.evidence_count}</td>
                  <td>
                    <span className="tag">{s.level}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* Quiz controls */}
          <div style={{ marginTop: 14, display: "flex", gap: 12, flexWrap: "wrap", alignItems: "end" }}>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <label className="muted" style={{ fontSize: 13 }}>
                Questions per skill
              </label>
              <input
                type="number"
                min={1}
                max={10}
                value={numPerSkill}
                onChange={(e) => setNumPerSkill(e.target.value)}
                style={{ width: 170 }}
              />
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <label className="muted" style={{ fontSize: 13 }}>
                Difficulty
              </label>
              <select value={difficulty} onChange={(e) => setDifficulty(e.target.value)} style={{ width: 170 }}>
                <option value="mixed">Mixed</option>
                <option value="easy">Easy</option>
                <option value="medium">Medium</option>
                <option value="hard">Hard</option>
              </select>
            </div>

            <button
              className="btn"
              onClick={handleGenerate}
              disabled={selectedSkills.length === 0}
              style={{ height: 38 }}
            >
              Generate Quiz (selected skills)
            </button>
          </div>
        </>
      )}
    </div>
  );
}
