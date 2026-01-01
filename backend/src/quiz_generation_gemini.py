import json
import re
from typing import List, Dict, Optional
from google import genai

# Import validation module
try:
    from src.quiz_validation import (
        validate_and_repair,
        _extract_json as extract_json_from_text
    )
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    def extract_json_from_text(text: str) -> str:
        """Fallback JSON extraction."""
        t = (text or "").strip()
        if t.startswith("{") and t.endswith("}"):
            return t
        m = re.search(r"\{.*\}", t, flags=re.DOTALL)
        if not m:
            raise ValueError("No JSON object found in Gemini output.")
        return m.group(0)

# Import diversity checking
try:
    from src.question_diversity import (
        check_duplicate_questions,
        filter_duplicates_from_questions,
        get_diversity_prompt_instruction
    )
    DIVERSITY_AVAILABLE = True
except ImportError:
    DIVERSITY_AVAILABLE = False
    def check_duplicate_questions(questions, similarity_threshold=0.85):
        return True, [], questions
    def filter_duplicates_from_questions(questions, similarity_threshold=0.85):
        return questions, []
    def get_diversity_prompt_instruction():
        return ""


def generate_mcqs_from_context(
    skill_key: str,
    context: str,
    n: int = 3,
    difficulty: str = "mixed",
    retrieved_chunk_ids: Optional[List[str]] = None
) -> dict:
    """
    Generate MCQ questions with robust validation.
    
    Args:
        skill_key: Skill name
        context: Context for question generation
        n: Number of questions to generate
        difficulty: Difficulty level
        retrieved_chunk_ids: Optional list of chunk IDs for citation validation
    
    Returns:
        {
          "skill_key": "...",
          "questions": [
             {
               "question": "...",
               "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
               "answer": "A",
               "explanation": "...",
               "evidence": [{"chunk_id": "...", "quote": "..."}]
             }
          ],
          "warnings": ["..."],  # Validation warnings/attempts
          "validated": true     # Whether validation passed
        }
    """
    client = genai.Client()
    
    chunk_ids_text = ""
    if retrieved_chunk_ids:
        chunk_ids_text = f"\nValid chunk IDs for citations: {', '.join(retrieved_chunk_ids[:10])}"

    # Get diversity instruction
    diversity_instruction = get_diversity_prompt_instruction() if DIVERSITY_AVAILABLE else ""
    
    prompt = f"""
You are an examiner generating multiple choice questions.

Create {n} multiple choice questions using ONLY the provided context.
Do not use outside knowledge. If the context is insufficient, return fewer questions.

SkillKey: {skill_key}
Difficulty: {difficulty}
NumberOfQuestions: {n}
{chunk_ids_text}

{diversity_instruction}

Context:
{context}

Return ONLY valid JSON in this EXACT schema:
{{
  "skill_key": "{skill_key}",
  "questions": [
    {{
      "question": "text (10-500 characters)",
      "options": {{"A":"...", "B":"...", "C":"...", "D":"..."}},
      "answer": "A|B|C|D",
      "explanation": "text (minimum 20 characters)",
      "evidence": [
        {{"chunk_id": "chunk_id_from_context", "quote": "relevant quote"}}
      ]
    }}
  ]
}}

IMPORTANT RULES:
- All options (A, B, C, D) must be unique
- Answer must be exactly A, B, C, or D
- Question text: 10-500 characters
- Explanation: minimum 20 characters
- Evidence chunk_ids must match those in context
- Return ONLY JSON, no markdown, no explanation
""".strip()

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    raw = resp.text or ""
    json_text = extract_json_from_text(raw)
    raw_json = json.loads(json_text)
    
    # Add default fields if missing
    if "questions" not in raw_json:
        raw_json["questions"] = []
    
    # Validate with repair and regeneration if available
    valid_questions = []
    all_warnings = []
    
    if VALIDATION_AVAILABLE and retrieved_chunk_ids:
        try:
            valid_questions, validation_warnings = validate_and_repair(
                raw_json=raw_json,
                skill=skill_key,
                context=context,
                retrieved_chunk_ids=retrieved_chunk_ids,
                max_repair_attempts=1,
                max_regeneration_attempts=1
            )
            all_warnings.extend(validation_warnings)
        except Exception as e:
            # If validation fails catastrophically, use original questions
            valid_questions = raw_json.get("questions", [])
            all_warnings.append(f"Validation error: {str(e)}")
    else:
        # Use original questions without validation
        valid_questions = raw_json.get("questions", [])
        if not VALIDATION_AVAILABLE:
            all_warnings.append("Validation not available")
    
    # Check for duplicate questions using diversity module
    if DIVERSITY_AVAILABLE and len(valid_questions) > 1:
        try:
            original_count = len(valid_questions)
            is_diverse, duplicate_warnings, unique_questions = check_duplicate_questions(
                valid_questions,
                similarity_threshold=0.85
            )
            
            if not is_diverse:
                # Remove duplicates
                removed_count = original_count - len(unique_questions)
                valid_questions = unique_questions
                all_warnings.extend(duplicate_warnings)
                if removed_count > 0:
                    all_warnings.append(
                        f"Removed {removed_count} duplicate question(s). "
                        "Consider regenerating with more diverse question types."
                    )
        except Exception as e:
            # If duplicate check fails, continue with original questions
            all_warnings.append(f"Duplicate check failed: {str(e)}")
    
    return {
        "skill_key": skill_key,
        "questions": valid_questions,
        "warnings": all_warnings,
        "validated": len(valid_questions) > 0
    }
