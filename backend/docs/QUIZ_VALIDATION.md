# Quiz Question Validation

Robust validation system for Gemini-generated quiz questions with automatic repair and regeneration.

## Overview

The validation pipeline ensures all generated questions meet quality standards before saving:

1. **Initial Validation**: Check JSON structure and all fields
2. **Repair Attempt**: Ask Gemini to fix validation errors (1 attempt)
3. **Regeneration**: Generate new questions with different chunks or reduced difficulty (1 attempt)

## Validation Checks

### 1. JSON Structure
- ✅ JSON parses correctly
- ✅ Has required fields (`questions` array)
- ✅ Questions array is not empty

### 2. Required Fields
Each question must have:
- `question`: Question text
- `options`: Object with A, B, C, D
- `answer`: Correct answer (A, B, C, or D)
- `explanation`: Explanation text
- `evidence`: Array of citations (optional but validated if present)

### 3. Answer Validation
- ✅ Must be exactly "A", "B", "C", or "D" (case-insensitive)

### 4. Options Validation
- ✅ All options (A, B, C, D) must exist
- ✅ All options must be non-empty
- ✅ Options must be unique (no duplicates)
- ✅ Option text max 200 characters

### 5. Question Text Validation
- ✅ Not empty
- ✅ Minimum 10 characters
- ✅ Maximum 500 characters

### 6. Explanation Validation
- ✅ Must be present
- ✅ Minimum 20 characters

### 7. Citation Validation
- ✅ Citations/evidence must not be empty (if present)
- ✅ `chunk_id` values must match retrieved chunk IDs
- ✅ Validates against actual chunk IDs from RAG retrieval

## Validation Flow

```
Gemini generates JSON
    ↓
Initial Validation
    ↓ (if fails)
Repair Attempt (1x)
    ↓ (if still fails)
Regeneration (1x)
    - Try with different context
    - Reduce difficulty if needed
    ↓ (if still fails)
Return valid questions (if any) + warnings
```

## Usage

### Basic Usage

```python
from src.quiz_validation import validate_and_repair

# After Gemini generates questions
valid_questions, warnings = validate_and_repair(
    raw_json=gemini_response,
    skill="SQL",
    context=retrieved_context,
    retrieved_chunk_ids=["sql_chunk_001", "sql_chunk_002", ...],
    max_repair_attempts=1,
    max_regeneration_attempts=1
)
```

### Integration in Generation

The validation is automatically integrated into `generate_mcqs_from_context`:

```python
from src.quiz_generation_gemini import generate_mcqs_from_context

result = generate_mcqs_from_context(
    skill_key="SQL",
    context=context,
    n=3,
    difficulty="medium",
    retrieved_chunk_ids=["sql_chunk_001", "sql_chunk_002"]
)

# Result includes:
# - result["questions"]: Validated questions
# - result["warnings"]: Validation warnings/attempts
# - result["validated"]: Whether validation passed
```

## Validation Examples

### Valid Question
```json
{
  "question": "What does SQL stand for?",
  "options": {
    "A": "Structured Query Language",
    "B": "Simple Query Language",
    "C": "Standard Query Language",
    "D": "Sequential Query Language"
  },
  "answer": "A",
  "explanation": "SQL stands for Structured Query Language...",
  "evidence": [
    {"chunk_id": "sql_chunk_001", "quote": "SQL (Structured Query Language)..."}
  ]
}
```

### Invalid Question (will trigger repair)
```json
{
  "question": "What is SQL?",  // Too short (< 10 chars)
  "options": {
    "A": "Language",
    "B": "Language",  // Duplicate
    "C": "Database",
    "D": "Tool"
  },
  "answer": "E",  // Invalid
  "explanation": "Short",  // Too short (< 20 chars)
  "evidence": [
    {"chunk_id": "invalid_chunk", "quote": "..."}  // Invalid chunk_id
  ]
}
```

## Repair Process

When validation fails, the system:

1. **Sends error message to Gemini** with:
   - List of all validation errors
   - Original JSON
   - Valid chunk IDs
   - Context (for reference)

2. **Asks Gemini to fix** all errors

3. **Re-validates** the repaired JSON

## Regeneration Process

If repair fails, the system:

1. **Reduces difficulty** if needed:
   - hard → medium
   - medium → easy
   - easy → easy (no change)

2. **Regenerates** with:
   - Same context (or could fetch different chunks)
   - Reduced difficulty
   - Explicit validation requirements in prompt

## Error Handling

### No Valid Questions After All Attempts
Returns empty list with warnings:
```python
{
    "questions": [],
    "warnings": [
        "Initial validation failed: 5 errors",
        "Repair failed: 3 errors remain",
        "Regeneration attempt 1 failed: 4 errors",
        "WARNING: All validation attempts failed"
    ],
    "validated": False
}
```

### Partial Success
Returns valid questions, logs warnings for invalid ones:
```python
{
    "questions": [valid_question_1, valid_question_2],
    "warnings": [
        "Question[0]: Answer must be A, B, C, or D",
        "Question[2]: Options must be unique"
    ],
    "validated": True
}
```

## Configuration

### Adjust Validation Strictness

```python
# More lenient (fewer repair attempts)
validate_and_repair(
    ...,
    max_repair_attempts=0,  # Skip repair
    max_regeneration_attempts=0  # Skip regeneration
)

# More strict (multiple attempts)
validate_and_repair(
    ...,
    max_repair_attempts=2,
    max_regeneration_attempts=2
)
```

### Adjust Length Limits

Edit `quiz_validation.py`:
```python
# In validate_question()
elif len(q_text) > 500:  # Change max length
elif len(q_text) < 10:   # Change min length
```

## Logging

Validation warnings are returned in the result:

```python
result = generate_mcqs_from_context(...)
for warning in result["warnings"]:
    print(f"⚠️  {warning}")
```

## File Locations

- **Validation module**: `backend/src/quiz_validation.py`
- **Gemini integration**: `backend/src/quiz_generation_gemini.py`
- **API integration**: `backend/src/api/main.py`

## Testing

Test validation manually:

```python
from src.quiz_validation import validate_question

# Test single question
is_valid, errors = validate_question(
    {
        "question": "What is SQL?",
        "options": {"A": "A", "B": "B", "C": "C", "D": "D"},
        "answer": "A",
        "explanation": "SQL is a database language..."
    },
    retrieved_chunk_ids=["sql_chunk_001"]
)

if not is_valid:
    print("Errors:", errors)
```

## Best Practices

1. **Always provide `retrieved_chunk_ids`** for citation validation
2. **Check warnings** in the result to understand what happened
3. **Monitor validation failures** - frequent failures may indicate:
   - Poor quality context
   - Need for better prompt engineering
   - Model issues

4. **Adjust difficulty** if regeneration is frequent:
   - Start with easier difficulty
   - Increase after validation succeeds

5. **Log validation results** for analysis:
   ```python
   if not result["validated"]:
       log_validation_failure(skill, result["warnings"])
   ```

