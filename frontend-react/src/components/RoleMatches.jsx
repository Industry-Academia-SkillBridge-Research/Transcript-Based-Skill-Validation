import { useState } from "react";

function RoleMatches({ studentId, data, onRolesLoaded, apiBase }) {
  const [loading, setLoading] = useState(false);
  const [localError, setLocalError] = useState(null);

  const handleLoadRoles = async () => {
    if (!studentId) {
      alert("Enter Student ID first.");
      return;
    }
    try {
      setLoading(true);
      setLocalError(null);
      const res = await fetch(
        `${apiBase}/students/${encodeURIComponent(studentId)}/roles`
      );
      if (!res.ok) {
        const text = await res.text();
        throw new Error("HTTP " + res.status + ": " + text);
      }
      const json = await res.json();
      onRolesLoaded && onRolesLoaded(json);
    } catch (err) {
      console.error(err);
      setLocalError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const roles = data?.roles || [];

  return (
    <section className="card">
      <h2>Step 4 · Job role matches</h2>
      <p className="muted">
        These matches consider your skills (transcript + quiz-adjusted) and job
        skill requirements mined from real postings.
      </p>
      <button
        className="secondary"
        onClick={handleLoadRoles}
        disabled={loading || !studentId}
      >
        {loading ? "Loading..." : "Refresh role matches"}
      </button>

      {localError && (
        <p style={{ color: "#b91c1c", marginTop: "0.5rem" }}>{localError}</p>
      )}

      {!roles.length ? (
        <p style={{ marginTop: "0.5rem" }} className="muted">
          No role matches loaded yet. Click "Refresh role matches".
        </p>
      ) : (
        <table style={{ marginTop: "0.75rem" }}>
          <thead>
            <tr>
              <th>Role</th>
              <th>Readiness</th>
              <th>Coverage</th>
              <th>Skills present</th>
              <th>Weak/missing</th>
            </tr>
          </thead>
          <tbody>
            {roles.map((r, idx) => (
              <tr key={idx}>
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
    </section>
  );
}

export default RoleMatches;
