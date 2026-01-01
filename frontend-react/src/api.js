// src/api.js
const API_BASE = (import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000").replace(/\/$/, "");
console.log("API_BASE =", API_BASE);

async function parseErrorResponse(res) {
  const text = await res.text();
  try {
    const data = JSON.parse(text);
    return data?.detail ? { detail: data.detail } : data;
  } catch {
    return { detail: text || `HTTP ${res.status}` };
  }
}

async function apiFetch(path, options = {}) {
  const url = `${API_BASE}${path.startsWith("/") ? "" : "/"}${path}`;

  try {
    const res = await fetch(url, options);

    if (!res.ok) {
      const err = await parseErrorResponse(res);
      const msg = err?.detail || `HTTP ${res.status}`;
      throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
    }

    const contentType = res.headers.get("content-type") || "";
    if (contentType.includes("application/json")) return res.json();
    return res.text();
  } catch (e) {
    const msg = String(e?.message || e);
    if (msg.includes("Failed to fetch")) {
      throw new Error(
        `Failed to reach API at ${API_BASE}. ` +
          `Check: backend running, CORS allows http://localhost:5173, and .env has VITE_API_BASE.`
      );
    }
    throw e;
  }
}

export function getSkills(studentId) {
  return apiFetch(`/students/${encodeURIComponent(studentId)}/skills`);
}

export function getRoles(studentId) {
  return apiFetch(`/students/${encodeURIComponent(studentId)}/roles`);
}

// NEW: prepare quiz from selected skills
// payload example:
// { selected_skills: ["Python","SQL"], num_questions_per_skill: 3, difficulty: "mixed" }
export function prepareQuiz(studentId, payload = {}) {
  return apiFetch(`/students/${encodeURIComponent(studentId)}/prepare-quiz`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function submitQuiz(studentId, responses) {
  return apiFetch(`/students/${encodeURIComponent(studentId)}/submit-quiz`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ responses }),
  });
}

export function uploadTranscript(studentId, file, regNo = "") {
  const form = new FormData();
  form.append("file", file);
  if (regNo) form.append("regno", regNo);

  return apiFetch(`/students/${encodeURIComponent(studentId)}/upload-transcript`, {
    method: "POST",
    body: form,
  });
}

// XAI (optional)
export function getSkillEvidence(studentId) {
  return apiFetch(`/students/${encodeURIComponent(studentId)}/xai/skills`);
}

export function getRoleEvidence(studentId, roleName) {
  const q = roleName ? `?role=${encodeURIComponent(roleName)}` : "";
  return apiFetch(`/students/${encodeURIComponent(studentId)}/xai/roles${q}`);
}
