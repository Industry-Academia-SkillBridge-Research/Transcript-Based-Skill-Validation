// src/api.js

const API_BASE = "http://127.0.0.1:8000";

async function jsonFetch(path, options = {}) {
  const res = await fetch(API_BASE + path, {
    headers: {
      ...(options.headers || {}),
    },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status}: ${text}`);
  }
  return res.json();
}

// from main dataset / fused pipeline
export function getSkills(studentId) {
  return jsonFetch(`/students/${studentId}/skills`, {
    headers: { "Content-Type": "application/json" },
  });
}

export function getRoles(studentId) {
  return jsonFetch(`/students/${studentId}/roles`, {
    headers: { "Content-Type": "application/json" },
  });
}

export function prepareQuiz(studentId) {
  return jsonFetch(`/students/${studentId}/prepare-quiz`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
}

export function submitQuiz(studentId, responses) {
  return jsonFetch(`/students/${studentId}/submit-quiz`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ responses }),
  });
}

// upload transcript → parsed → skills (single student)
export async function uploadTranscript(studentId, regno, file) {
  const form = new FormData();
  form.append("file", file);
  if (regno) {
    form.append("regno", regno);
  }
  const res = await fetch(
    `${API_BASE}/students/${encodeURIComponent(studentId)}/upload-transcript`,
    {
      method: "POST",
      body: form,
    }
  );
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status}: ${text}`);
  }
  return res.json();
}
