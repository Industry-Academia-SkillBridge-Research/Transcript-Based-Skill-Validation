# Quiz Storage and Serving

## Overview

When a user clicks "Generate Quiz", the system:

1. **Creates a unique quiz_id** (UUID)
2. **For each selected skill**:
   - Retrieves chunks from corpus
   - Generates N questions
   - Validates questions
3. **Saves quiz to files**:
   - `output/quizzes/{quiz_id}_questions.json` - Questions with answers
   - `output/quizzes/{quiz_id}_meta.json` - Metadata
4. **Returns to frontend**:
   - `quiz_id`
   - Questions (without answers)
   - Time limit

## File Structure

### Questions File: `{quiz_id}_questions.json`

Contains questions WITH answers (for backend use only):

```json
[
  {
    "QuestionID": 100001,
    "SelectedSkill": "SQL",
    "Skill": "SQL",
    "Difficulty": "medium",
    "QuestionText": "What does SQL stand for?",
    "OptionA": "Structured Query Language",
    "OptionB": "Simple Query Language",
    "OptionC": "Standard Query Language",
    "OptionD": "Sequential Query Language",
    "CorrectOption": "A",
    "Explanation": "SQL stands for Structured Query Language...",
    "Source": "gemini"
  }
]
```

### Metadata File: `{quiz_id}_meta.json`

Contains quiz metadata:

```json
{
  "quiz_id": "550e8400-e29b-41d4-a716-446655440000",
  "student_id": "IT21013928",
  "selected_skills": ["SQL", "Python"],
  "expanded_skills": ["SQL", "Python"],
  "num_questions": 6,
  "num_questions_per_skill": 3,
  "difficulty": "mixed",
  "time_limit_minutes": 12,
  "time_limit_seconds": 720,
  "created_time": "2024-01-15T10:30:00",
  "answer_key": {
    "100001": "A",
    "100002": "B",
    ...
  }
}
```

## API Endpoints

### 1. Generate Quiz (POST)

**Endpoint**: `/students/{student_id}/prepare-quiz`

**Request**:
```json
{
  "selected_skills": ["SQL", "Python"],
  "num_questions_per_skill": 3,
  "difficulty": "mixed"
}
```

**Response** (no answers):
```json
{
  "quiz_id": "550e8400-e29b-41d4-a716-446655440000",
  "student_id": "IT21013928",
  "selected_skills": ["SQL", "Python"],
  "num_questions": 6,
  "time_limit_minutes": 12,
  "time_limit_seconds": 720,
  "questions": [
    {
      "QuestionID": 100001,
      "QuestionText": "...",
      "OptionA": "...",
      "OptionB": "...",
      "OptionC": "...",
      "OptionD": "...",
      "Explanation": "..."
      // NO CorrectOption field!
    }
  ]
}
```

### 2. Get Quiz (GET)

**Endpoint**: `/quizzes/{quiz_id}`

**Response**: Same as generate quiz response (questions without answers)

### 3. Submit Quiz (POST)

**Endpoint**: `/students/{student_id}/submit-quiz`

**Request**:
```json
{
  "responses": [
    {
      "question_id": 100001,
      "selected_option": "A",
      "response_time_seconds": 45
    }
  ],
  "quiz_id": "550e8400-e29b-41d4-a716-446655440000"  // Optional
}
```

**Note**: If `quiz_id` is provided, uses that quiz's answer key. Otherwise falls back to student_id-based answer key.

## Time Limit Calculation

- **Default**: 2 minutes per question
- **Minimum**: 10 minutes
- **Formula**: `max(10, num_questions * 2)` minutes

Examples:
- 3 questions → 10 minutes (minimum)
- 5 questions → 10 minutes
- 6 questions → 12 minutes
- 10 questions → 20 minutes

## Security: No Answers in Frontend

✅ **Questions sent to frontend**: NO `CorrectOption` field
✅ **Questions stored in backend**: Include `CorrectOption` field
✅ **Answer key**: Stored in metadata file, never sent to frontend

## Workflow

```
User clicks "Generate Quiz"
    ↓
Backend:
  1. Create quiz_id (UUID)
  2. For each skill:
     - Retrieve chunks (RAG)
     - Generate questions (Gemini)
     - Validate questions
  3. Save to files:
     - {quiz_id}_questions.json (with answers)
     - {quiz_id}_meta.json (metadata)
  4. Return to frontend:
     - quiz_id
     - questions (NO answers)
     - time_limit
    ↓
Frontend:
  - Store quiz_id
  - Display questions
  - Start timer
  - On submit: send quiz_id + responses
    ↓
Backend:
  - Load answer key from {quiz_id}_meta.json
  - Score responses
  - Return results
```

## File Locations

- **Quiz questions**: `backend/output/quizzes/{quiz_id}_questions.json`
- **Quiz metadata**: `backend/output/quizzes/{quiz_id}_meta.json`
- **Answer keys** (legacy): `backend/output/quiz_answer_key_{student_id}.json`

## Backward Compatibility

The system maintains backward compatibility:
- Still saves `quiz_answer_key_{student_id}.json` for old submissions
- `submit-quiz` accepts optional `quiz_id` in payload
- Falls back to student_id-based answer key if quiz_id not provided

