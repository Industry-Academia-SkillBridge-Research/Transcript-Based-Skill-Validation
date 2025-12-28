// src/components/SkillTable.jsx
export function SkillTable({ title, data, error }) {
  return (
    <div className="card">
      <h2>{title}</h2>
      {error && <div className="error-box">{error}</div>}
      {!data || !data.skills || data.skills.length === 0 ? (
        <div className="muted">No skills found.</div>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Skill</th>
              <th>Score</th>
              <th>Evidence</th>
              <th>Level</th>
            </tr>
          </thead>
          <tbody>
            {data.skills.map((s) => (
              <tr key={s.skill}>
                <td>{s.skill}</td>
                <td>{s.score.toFixed(3)}</td>
                <td>{s.evidence_count}</td>
                <td>
                  <span className="tag">{s.level}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
