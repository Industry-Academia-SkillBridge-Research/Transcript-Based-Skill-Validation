import os
import re
import json
from typing import List, Dict, Any

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chunk_text(text: str, max_words: int = 350) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i + max_words]))
    return chunks

def simple_retrieve(skill: str, query: str, top_k: int = 4) -> List[Dict[str, str]]:
    """
    Very simple retrieval: split skill file into chunks and rank by keyword hits.
    """
    path = os.path.join("knowledge_base", f"{skill.lower()}.txt")
    if not os.path.exists(path):
        return []

    raw = open(path, "r", encoding="utf-8").read()
    chunks = chunk_text(raw)

    q_terms = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
    scored = []
    for idx, ch in enumerate(chunks):
        c_terms = set(re.findall(r"[a-zA-Z0-9]+", ch.lower()))
        score = len(q_terms.intersection(c_terms))
        scored.append((score, idx, ch))

    scored.sort(reverse=True, key=lambda x: x[0])
    top = scored[:top_k]

    out = []
    for score, idx, ch in top:
        out.append({"chunk_id": f"{skill.lower()}_{idx:03d}", "text": ch})
    return out

def build_prompt(skill: str, difficulty: str, n: int, contexts: List[Dict[str, str]]) -> str:
    context_block = "\n\n".join(
        [f"[{c['chunk_id']}]\n{c['text']}" for c in contexts]
    )

    return f"""
You are generating MCQ questions for skill: {skill}.
Difficulty: {difficulty}.
You MUST use only the facts in the CONTEXT below.
If the CONTEXT is insufficient, output an empty questions array.

Return STRICT JSON only, with this schema:

{{
  "questions": [
    {{
      "skill": "{skill}",
      "difficulty": "{difficulty}",
      "question": "string",
      "options": {{"A":"...", "B":"...", "C":"...", "D":"..."}},
      "answer": "A|B|C|D",
      "explanation": "short explanation using only context",
      "evidence": [
        {{"chunk_id":"...", "quote":"short quote from context"}}
      ]
    }}
  ]
}}

Rules:
- Exactly {n} questions if possible.
- Each question must include at least 1 evidence item with a quote that appears in CONTEXT.
- Do NOT use outside knowledge.
- Do NOT include markdown. Only JSON.

CONTEXT:
{context_block}
""".strip()

def validate_questions(data: Dict[str, Any], contexts: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Light validation: ensure evidence quote appears in provided context.
    """
    allowed_text = "\n".join([c["text"] for c in contexts])

    out = []
    for q in data.get("questions", []):
        if not isinstance(q, dict):
            continue
        if q.get("answer") not in ["A", "B", "C", "D"]:
            continue
        ev = q.get("evidence") or []
        if not ev or not isinstance(ev, list):
            continue
        quote_ok = False
        for e in ev:
            quote = (e.get("quote") or "").strip()
            if quote and quote in allowed_text:
                quote_ok = True
        if not quote_ok:
            continue
        out.append(q)
    return out

def generate_mcqs_for_skill(skill: str, difficulty: str, n: int) -> List[Dict[str, Any]]:
    query = f"{skill} {difficulty} MCQ"
    contexts = simple_retrieve(skill, query, top_k=6)
    if not contexts:
        return []

    prompt = build_prompt(skill, difficulty, n, contexts)

    # Use Responses API style through SDK; keep it simple: one text output
    resp = client.responses.create(
        model="gpt-5-mini",
        input=prompt,
    )

    text = resp.output_text.strip()

    # parse JSON safely
    try:
        data = json.loads(text)
    except Exception:
        return []

    valid = validate_questions(data, contexts)
    return valid
