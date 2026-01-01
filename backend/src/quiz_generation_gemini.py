import json
import re
from google import genai


def _extract_json(text: str) -> str:
    """
    Gemini sometimes wraps JSON in text. This extracts the first JSON object.
    """
    t = (text or "").strip()
    if t.startswith("{") and t.endswith("}"):
        return t
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in Gemini output.")
    return m.group(0)


def generate_mcqs_from_context(skill_key: str, context: str, n: int = 3, difficulty: str = "mixed") -> dict:
    """
    Returns:
      {
        "skill_key": "...",
        "questions": [
           {"question": "...", "options": {"A": "...", ...}, "answer": "A"}
        ]
      }
    """
    client = genai.Client()

    prompt = f"""
You are an examiner.

Create multiple choice questions using ONLY the provided context.
Do not use outside knowledge. If the context is insufficient, return fewer questions.

SkillKey: {skill_key}
Difficulty: {difficulty}
NumberOfQuestions: {n}

Context:
{context}

Return ONLY valid JSON in this EXACT schema:
{{
  "skill_key": "{skill_key}",
  "questions": [
    {{
      "question": "text",
      "options": {{"A":"...","B":"...","C":"...","D":"..."}},
      "answer": "A"
    }}
  ]
}}
""".strip()

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    raw = resp.text or ""
    json_text = _extract_json(raw)
    return json.loads(json_text)
