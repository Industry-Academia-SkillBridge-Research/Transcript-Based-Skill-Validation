// src/components/RoleTable.jsx
export function RoleTable({ title, data, error }) {
  return (
    <div className="card">
      <h2>{title}</h2>
      {error && <div className="error-box">{error}</div>}
      {!data || !data.roles || data.roles.length === 0 ? (
        <div className="muted">No role matches found.</div>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Role</th>
              <th>Readiness</th>
              <th>Coverage</th>
              <th>Skills present</th>
              <th>Weak / missing</th>
            </tr>
          </thead>
          <tbody>
            {data.roles.map((r) => (
              <tr key={r.role_name}>
                <td>{r.role_name}</td>
                <td>{(r.readiness_score * 100).toFixed(1)}%</td>
                <td>{(r.coverage * 100).toFixed(1)}%</td>
                <td>
                  {r.num_skills_present}/{r.num_skills}
                </td>
                <td>{r.num_weak_or_missing}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
