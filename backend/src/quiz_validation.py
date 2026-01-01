"""
quiz_validation.py

Robust validation for Gemini-generated quiz questions.

Features:
- JSON parsing validation
- Field completeness checks
- Option uniqueness validation
- Question length validation
- Citation validation against retrieved chunk IDs
- Automatic repair via Gemini
- Fallback regeneration with different chunks or reduced difficulty
"""

import json
import re
from typing import Dict, List, Optional, Tuple, Any
from google import genai


# -----------------------------
# Validation Functions
# -----------------------------

class ValidationError(Exception):
    """Custom exception for validation errors."""
    def __init__(self, message: str, errors: List[str]):
        super().__init__(message)
        self.errors = errors


def _extract_json(text: str) -> str:
    """
    Extract JSON from Gemini output (may be wrapped in text).
    """
    t = (text or "").strip()
    if t.startswith("{") and t.endswith("}"):
        return t
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in Gemini output.")
    return m.group(0)


def validate_json_structure(data: Any) -> Tuple[bool, List[str]]:
    """
    Validate that JSON parses correctly and has required structure.
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    if not isinstance(data, dict):
        errors.append("Root must be a JSON object")
        return False, errors
    
    if "questions" not in data:
        errors.append("Missing 'questions' field")
        return False, errors
    
    if not isinstance(data["questions"], list):
        errors.append("'questions' must be an array")
        return False, errors
    
    if len(data["questions"]) == 0:
        errors.append("'questions' array is empty")
        return False, errors
    
    return True, errors


def validate_question(question: Dict[str, Any], retrieved_chunk_ids: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate a single question object.
    
    Args:
        question: Question dictionary
        retrieved_chunk_ids: List of valid chunk IDs from retrieval
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    # Required fields
    required_fields = ["question", "options", "answer", "explanation"]
    for field in required_fields:
        if field not in question:
            errors.append(f"Missing required field: '{field}'")
    
    # Validate question text
    if "question" in question:
        q_text = str(question["question"]).strip()
        if not q_text:
            errors.append("Question text is empty")
        elif len(q_text) > 500:
            errors.append(f"Question text too long ({len(q_text)} chars, max 500)")
        elif len(q_text) < 10:
            errors.append(f"Question text too short ({len(q_text)} chars, min 10)")
    
    # Validate options
    if "options" in question:
        opts = question["options"]
        if not isinstance(opts, dict):
            errors.append("'options' must be an object")
        else:
            # Check all required options exist
            required_opts = ["A", "B", "C", "D"]
            missing_opts = [opt for opt in required_opts if opt not in opts]
            if missing_opts:
                errors.append(f"Missing options: {', '.join(missing_opts)}")
            
            # Check option values are not empty
            for opt in required_opts:
                if opt in opts:
                    opt_text = str(opts[opt]).strip()
                    if not opt_text:
                        errors.append(f"Option {opt} is empty")
                    elif len(opt_text) > 200:
                        errors.append(f"Option {opt} too long ({len(opt_text)} chars, max 200)")
            
            # Check options are unique (no duplicates)
            opt_values = [str(opts.get(opt, "")).strip().lower() for opt in required_opts if opt in opts]
            if len(opt_values) != len(set(opt_values)):
                errors.append("Options must be unique (found duplicates)")
    
    # Validate correct answer
    if "answer" in question:
        answer = str(question["answer"]).strip().upper()
        if answer not in ["A", "B", "C", "D"]:
            errors.append(f"Answer must be A, B, C, or D (got '{answer}')")
    
    # Validate explanation
    if "explanation" in question:
        explanation = str(question["explanation"]).strip()
        if not explanation:
            errors.append("Explanation is empty")
        elif len(explanation) < 20:
            errors.append(f"Explanation too short ({len(explanation)} chars, min 20)")
    
    # Validate citations/evidence (if present)
    if "evidence" in question or "citations" in question:
        evidence = question.get("evidence") or question.get("citations", [])
        if not isinstance(evidence, list):
            errors.append("Evidence/citations must be an array")
        elif len(evidence) == 0:
            errors.append("Evidence/citations array is empty")
        else:
            # Check that chunk IDs in citations match retrieved chunk IDs
            for idx, item in enumerate(evidence):
                if isinstance(item, dict):
                    chunk_id = item.get("chunk_id") or item.get("chunkId") or item.get("chunk")
                    if chunk_id:
                        if retrieved_chunk_ids and chunk_id not in retrieved_chunk_ids:
                            errors.append(
                                f"Evidence[{idx}]: chunk_id '{chunk_id}' not in retrieved chunks. "
                                f"Valid: {retrieved_chunk_ids[:3]}..."
                            )
                elif isinstance(item, str):
                    if retrieved_chunk_ids and item not in retrieved_chunk_ids:
                        errors.append(
                            f"Evidence[{idx}]: chunk_id '{item}' not in retrieved chunks"
                        )
    
    return len(errors) == 0, errors


def validate_questions_batch(
    data: Dict[str, Any],
    retrieved_chunk_ids: List[str]
) -> Tuple[bool, List[str], List[Dict[str, Any]]]:
    """
    Validate all questions in a batch.
    
    Args:
        data: JSON data from Gemini
        retrieved_chunk_ids: List of valid chunk IDs from retrieval
    
    Returns:
        (all_valid, all_errors, valid_questions)
    """
    all_errors = []
    valid_questions = []
    
    # Validate structure
    struct_valid, struct_errors = validate_json_structure(data)
    if not struct_valid:
        return False, struct_errors, []
    
    # Validate each question
    for idx, question in enumerate(data["questions"]):
        if not isinstance(question, dict):
            all_errors.append(f"Question[{idx}]: Not a valid object")
            continue
        
        is_valid, errors = validate_question(question, retrieved_chunk_ids)
        if is_valid:
            valid_questions.append(question)
        else:
            error_msgs = [f"Question[{idx}]: {err}" for err in errors]
            all_errors.extend(error_msgs)
    
    all_valid = len(all_errors) == 0 and len(valid_questions) > 0
    
    if len(valid_questions) == 0:
        all_errors.append("No valid questions found after validation")
    
    return all_valid, all_errors, valid_questions


# -----------------------------
# Repair Function
# -----------------------------

def repair_questions(
    original_json: Dict[str, Any],
    validation_errors: List[str],
    skill: str,
    context: str,
    retrieved_chunk_ids: List[str]
) -> Dict[str, Any]:
    """
    Ask Gemini to repair invalid questions.
    
    Args:
        original_json: Original JSON from Gemini
        validation_errors: List of validation error messages
        skill: Skill name
        context: Context used for generation
        retrieved_chunk_ids: Valid chunk IDs
    
    Returns:
        Repaired JSON data
    """
    client = genai.Client()
    
    errors_text = "\n".join(f"- {err}" for err in validation_errors)
    chunk_ids_text = ", ".join(retrieved_chunk_ids[:10])  # Show first 10
    
    prompt = f"""
You generated quiz questions, but they failed validation. Please fix them.

VALIDATION ERRORS:
{errors_text}

SKILL: {skill}
VALID CHUNK IDs (use these in citations): {chunk_ids_text}

ORIGINAL JSON (with errors):
{json.dumps(original_json, indent=2)}

CONTEXT (for reference):
{context[:2000]}...

INSTRUCTIONS:
1. Fix ALL validation errors listed above
2. Ensure 'answer' is exactly A, B, C, or D
3. Ensure all options (A, B, C, D) are unique and non-empty
4. Question text must be 10-500 characters
5. Explanation must be at least 20 characters
6. Citations/evidence must use chunk_ids from the valid list above
7. Return ONLY valid JSON, no markdown, no explanation

Return the CORRECTED JSON in this exact schema:
{{
  "questions": [
    {{
      "question": "text (10-500 chars)",
      "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
      "answer": "A|B|C|D",
      "explanation": "text (min 20 chars)",
      "evidence": [
        {{"chunk_id": "must_match_valid_list", "quote": "..."}}
      ]
    }}
  ]
}}
""".strip()
    
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        raw = resp.text or ""
        json_text = _extract_json(raw)
        repaired = json.loads(json_text)
        return repaired
    except Exception as e:
        raise ValueError(f"Repair failed: {str(e)}")


# -----------------------------
# Regeneration Function
# -----------------------------

def regenerate_with_different_chunks(
    skill: str,
    difficulty: str,
    n: int,
    context: str,
    retrieved_chunk_ids: List[str],
    attempt: int = 1
) -> Dict[str, Any]:
    """
    Regenerate questions with different context or reduced difficulty.
    
    Args:
        skill: Skill name
        difficulty: Current difficulty (may be reduced)
        n: Number of questions to generate
        context: Original context
        retrieved_chunk_ids: Original chunk IDs
        attempt: Attempt number (reduces difficulty after attempt 1)
    
    Returns:
        New JSON data
    """
    client = genai.Client()
    
    # Reduce difficulty if this is a retry
    if attempt > 1:
        difficulty_map = {
            "hard": "medium",
            "medium": "easy",
            "easy": "easy",
            "mixed": "medium"
        }
        difficulty = difficulty_map.get(difficulty.lower(), "medium")
    
    prompt = f"""
You are generating MCQ questions. Previous attempts failed validation.
This is attempt #{attempt + 1}.

SKILL: {skill}
DIFFICULTY: {difficulty} (may have been reduced from original)
NUMBER OF QUESTIONS: {n}

CONTEXT:
{context[:3000]}...

IMPORTANT:
- Generate exactly {n} questions
- Ensure 'answer' is exactly A, B, C, or D
- All options must be unique
- Question: 10-500 characters
- Explanation: minimum 20 characters
- Use chunk_ids from context markers [chunk_id] if present

Return ONLY valid JSON:
{{
  "questions": [
    {{
      "question": "text",
      "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
      "answer": "A|B|C|D",
      "explanation": "text",
      "evidence": [{{"chunk_id": "...", "quote": "..."}}]
    }}
  ]
}}
""".strip()
    
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        raw = resp.text or ""
        json_text = _extract_json(raw)
        regenerated = json.loads(json_text)
        return regenerated
    except Exception as e:
        raise ValueError(f"Regeneration failed: {str(e)}")


# -----------------------------
# Main Validation Pipeline
# -----------------------------

def validate_and_repair(
    raw_json: Dict[str, Any],
    skill: str,
    context: str,
    retrieved_chunk_ids: List[str],
    max_repair_attempts: int = 1,
    max_regeneration_attempts: int = 1
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Complete validation pipeline with repair and regeneration.
    
    Args:
        raw_json: Raw JSON from Gemini
        skill: Skill name
        context: Context used for generation
        retrieved_chunk_ids: Valid chunk IDs from retrieval
        max_repair_attempts: Maximum repair attempts (default 1)
        max_regeneration_attempts: Maximum regeneration attempts (default 1)
    
    Returns:
        (valid_questions, warning_messages)
    """
    warnings = []
    current_data = raw_json
    
    # Step 1: Initial validation
    all_valid, errors, valid_questions = validate_questions_batch(
        current_data, retrieved_chunk_ids
    )
    
    if all_valid:
        warnings.append(f"Initial validation passed: {len(valid_questions)} questions")
        return valid_questions, warnings
    
    warnings.append(f"Initial validation failed: {len(errors)} errors")
    
    # Step 2: Repair attempt
    if max_repair_attempts > 0:
        try:
            warnings.append("Attempting repair...")
            repaired_data = repair_questions(
                current_data,
                errors,
                skill,
                context,
                retrieved_chunk_ids
            )
            
            # Validate repaired data
            all_valid, repair_errors, valid_questions = validate_questions_batch(
                repaired_data, retrieved_chunk_ids
            )
            
            if all_valid:
                warnings.append(f"Repair successful: {len(valid_questions)} valid questions")
                return valid_questions, warnings
            else:
                warnings.append(f"Repair failed: {len(repair_errors)} errors remain")
                current_data = repaired_data
                errors = repair_errors
        except Exception as e:
            warnings.append(f"Repair exception: {str(e)}")
    
    # Step 3: Regeneration attempts
    difficulty = "mixed"  # Default, could be passed as parameter
    for attempt in range(max_regeneration_attempts):
        try:
            warnings.append(f"Attempting regeneration (attempt {attempt + 1})...")
            regenerated_data = regenerate_with_different_chunks(
                skill=skill,
                difficulty=difficulty,
                n=len(raw_json.get("questions", [])),
                context=context,
                retrieved_chunk_ids=retrieved_chunk_ids,
                attempt=attempt
            )
            
            # Validate regenerated data
            all_valid, regen_errors, valid_questions = validate_questions_batch(
                regenerated_data, retrieved_chunk_ids
            )
            
            if all_valid:
                warnings.append(f"Regeneration successful: {len(valid_questions)} valid questions")
                return valid_questions, warnings
            else:
                warnings.append(f"Regeneration attempt {attempt + 1} failed: {len(regen_errors)} errors")
        except Exception as e:
            warnings.append(f"Regeneration attempt {attempt + 1} exception: {str(e)}")
    
    # Final: Return what we have (may be empty)
    warnings.append(f"WARNING: All validation attempts failed. Returning {len(valid_questions)} valid questions (if any)")
    return valid_questions, warnings

