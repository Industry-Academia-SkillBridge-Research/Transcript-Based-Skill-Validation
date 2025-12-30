function SkillProfile({ data }) {
  if (!data || !data.skills || !data.skills.length) {
    return (
      <section className="card">
        <h2>Step 2 · Skill profile from transcript</h2>
        <p className="muted">
          No skills to show yet. Upload a transcript or load skills from DB.
        </p>
      </section>
    );
  }

  const skills = data.skills;
  const rows = skills.map((s, idx) => (
    <tr key={idx}>
      <td>{s.skill}</td>
      <td>{s.score.toFixed(3)}</td>
      <td>{s.evidence_count}</td>
      <td>
        <span className="tag">{s.level}</span>
      </td>
    </tr>
  ));

  return (
    <section className="card">
      <h2>Step 2 · Skill profile inferred from transcript</h2>
      <p className="muted">
        These are the skills the system believes you learned based on your
        modules and grades.
      </p>
      <table>
        <thead>
          <tr>
            <th>Skill</th>
            <th>Score</th>
            <th>Evidence</th>
            <th>Level</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </section>
  );
}

export default SkillProfile;
