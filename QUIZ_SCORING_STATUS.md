# Quiz Scoring - Implementation Status

## ✅ REQUIREMENT 7: Quiz Scoring (Answers → Verified Skill Score)

**Goal**: Score quiz submissions and calculate verified skill proficiency from quiz performance.

**Input**: Student submits question_id, selected_option, response_time_seconds

**Output**: Overall accuracy, per-skill accuracy, detailed results per question

---

## Status: ✅ **FULLY IMPLEMENTED**

### ✅ Input Format
**Status**: ✅ **COMPLETE**

**Student Submission:**
```json
{
  "responses": [
    {
      "question_id": 100001,
      "selected_option": "A",
      "response_time_seconds": 45.5
    },
    ...
  ],
  "quiz_id": "550e8400-e29b-41d4-a716-446655440000",
  "session_token": "...",  // Optional
  "violations": [...]       // Optional
}
```

**Files:**
- `backend/src/api/main.py` - `submit_quiz` endpoint accepts this format

---

### ✅ Scoring Output
**Status**: ✅ **COMPLETE**

#### 1. Overall Accuracy
- ✅ Calculated as: `correct / total`
- ✅ Returned in response
- ✅ Displayed on frontend

#### 2. Per-Skill Accuracy
- ✅ Calculated per skill from quiz performance
- ✅ Includes:
  - Total questions per skill
  - Correct answers per skill
  - Accuracy percentage per skill
  - Average response time per skill
- ✅ Returned in response as `per_skill` array
- ✅ Displayed on frontend

#### 3. Detailed Results Per Question
- ✅ One result per answered question
- ✅ Includes:
  - Question ID, text, skill, difficulty
  - Selected option vs. correct option
  - Is correct (boolean)
  - Response time in seconds
- ✅ Returned in response as `per_question` array
- ✅ Displayed on frontend in review mode

**Response Format:**
```json
{
  "student_id": "IT21013928",
  "quiz_id": "550e8400-e29b-41d4-a716-446655440000",
  "total_answered": 9,
  "correct": 7,
  "accuracy": 0.7778,
  "per_question": [
    {
      "question_id": 100001,
      "question_text": "What is SQL?",
      "skill": "SQL & Relational Databases",
      "difficulty": "medium",
      "selected_option": "A",
      "correct_option": "A",
      "is_correct": true,
      "response_time_seconds": 45.5
    },
    ...
  ],
  "per_skill": [
    {
      "skill": "SQL & Relational Databases",
      "total_questions": 3,
      "correct_answers": 2,
      "accuracy": 0.6667,
      "avg_response_time_seconds": 42.3
    },
    ...
  ]
}
```

**Files:**
- `backend/src/api/main.py` - Scoring logic in `submit_quiz` endpoint

---

### ✅ File Outputs
**Status**: ✅ **COMPLETE**

#### 1. `output/quiz_results_scored.csv`
**Status**: ✅ **IMPLEMENTED**

**Format:** One row per answered question

**Columns:**
- `StudentID` - Student identifier
- `QuizID` - Quiz identifier
- `QuestionID` - Question identifier
- `QuestionText` - Question text
- `Skill` - Skill tested
- `Difficulty` - Question difficulty
- `SelectedOption` - Student's selected option (A/B/C/D)
- `CorrectOption` - Correct answer (A/B/C/D)
- `IsCorrect` - Boolean (True/False)
- `ResponseTimeSeconds` - Time taken to answer (seconds)

**Example:**
```csv
StudentID,QuizID,QuestionID,QuestionText,Skill,Difficulty,SelectedOption,CorrectOption,IsCorrect,ResponseTimeSeconds
IT21013928,550e8400-...,100001,"What is SQL?","SQL & Relational Databases",medium,A,A,True,45.5
IT21013928,550e8400-...,100002,"Which SQL keyword...","SQL & Relational Databases",medium,B,C,False,32.1
```

**Implementation:**
- Saved automatically on quiz submission
- Appended to existing file (accumulates results)
- No duplicates (newest results appended)

#### 2. `output/skill_quiz_updates.csv`
**Status**: ✅ **IMPLEMENTED**

**Format:** One row per (StudentID, QuizID, Skill)

**Columns:**
- `StudentID` - Student identifier
- `QuizID` - Quiz identifier
- `Skill` - Skill name
- `NumQuestions` - Number of questions answered for this skill
- `NumCorrect` - Number of correct answers
- `QuizProficiency` - Verified proficiency score (0-1) = accuracy
- `AvgResponseTimeSeconds` - Average time per question (seconds)

**Example:**
```csv
StudentID,QuizID,Skill,NumQuestions,NumCorrect,QuizProficiency,AvgResponseTimeSeconds
IT21013928,550e8400-...,"SQL & Relational Databases",3,2,0.6667,42.3
IT21013928,550e8400-...,"Python Programming",3,3,1.0,38.7
IT21013928,550e8400-...,"Data Structures",3,2,0.6667,51.2
```

**Implementation:**
- Saved automatically on quiz submission
- Replaces old entries for same (StudentID, QuizID) combination
- Appends new entries for different quizzes

**Files:**
- `backend/src/api/main.py` - File saving logic in `submit_quiz` endpoint

---

### ✅ Scoring Logic
**Status**: ✅ **COMPLETE**

#### Per-Question Scoring:
1. Compare `selected_option` with `correct_option` (case-insensitive)
2. Mark as correct if match and option is valid (A/B/C/D)
3. Track response time for analytics

#### Per-Skill Aggregation:
1. Group questions by skill
2. Count total questions per skill
3. Count correct answers per skill
4. Calculate accuracy: `correct / total`
5. Calculate average response time

#### Overall Accuracy:
1. Sum all correct answers
2. Divide by total questions answered
3. Return as decimal (0.0 to 1.0)

---

## 📊 Data Flow

```
1. Student submits quiz
   ↓
2. Backend receives responses:
   - question_id
   - selected_option
   - response_time_seconds
   ↓
3. Backend loads answer key from quiz metadata
   ↓
4. Backend scores each response:
   - Compare selected vs correct option
   - Mark is_correct
   - Track skill and response time
   ↓
5. Backend calculates:
   - Overall accuracy
   - Per-skill accuracy
   - Detailed results per question
   ↓
6. Backend saves to CSV files:
   - quiz_results_scored.csv (detailed)
   - skill_quiz_updates.csv (aggregated)
   ↓
7. Backend returns results to frontend
   ↓
8. Frontend displays:
   - Overall score
   - Per-skill breakdown
   - Detailed question review
```

---

## 📁 File Structure

### `output/quiz_results_scored.csv`
**Purpose**: Detailed record of all quiz attempts (one row per question)

**Usage:**
- Research analysis
- Audit trail
- Detailed performance tracking

### `output/skill_quiz_updates.csv`
**Purpose**: Aggregated skill proficiency from quiz validation

**Usage:**
- Skill profile fusion (combine with transcript scores)
- Job role matching (verified skills)
- Research analysis

**Note**: This file is used by `skill_profile_fusion.py` to combine transcript-based scores with quiz-verified proficiency.

---

## ✅ Verification Checklist

**Input:**
- [x] Accepts question_id in responses
- [x] Accepts selected_option in responses
- [x] Accepts response_time_seconds in responses

**Scoring:**
- [x] Compares selected vs correct option
- [x] Calculates overall accuracy
- [x] Calculates per-skill accuracy
- [x] Tracks response times
- [x] Generates detailed per-question results

**Output:**
- [x] Returns overall accuracy
- [x] Returns per-skill accuracy array
- [x] Returns detailed per-question results
- [x] Saves to quiz_results_scored.csv
- [x] Saves to skill_quiz_updates.csv

**File Format:**
- [x] quiz_results_scored.csv has all required columns
- [x] skill_quiz_updates.csv has all required columns
- [x] Files append new results (don't overwrite)
- [x] Old entries for same quiz are replaced in skill_quiz_updates.csv

---

## 📝 Example Scoring Scenario

**Input:**
- Student: IT21013928
- Quiz: 9 questions (3 skills, 3 questions each)
- Responses:
  - SQL: 2/3 correct
  - Python: 3/3 correct
  - Data Structures: 2/3 correct

**Scoring Output:**

**Overall:**
- Total: 9
- Correct: 7
- Accuracy: 0.7778 (77.78%)

**Per-Skill:**
```json
[
  {
    "skill": "SQL & Relational Databases",
    "total_questions": 3,
    "correct_answers": 2,
    "accuracy": 0.6667,
    "avg_response_time_seconds": 42.3
  },
  {
    "skill": "Python Programming",
    "total_questions": 3,
    "correct_answers": 3,
    "accuracy": 1.0,
    "avg_response_time_seconds": 38.7
  },
  {
    "skill": "Data Structures",
    "total_questions": 3,
    "correct_answers": 2,
    "accuracy": 0.6667,
    "avg_response_time_seconds": 51.2
  }
]
```

**Files Saved:**

**quiz_results_scored.csv:**
- 9 rows (one per question)

**skill_quiz_updates.csv:**
- 3 rows (one per skill)

---

## 🔄 Integration with Skill Profile Fusion

The `skill_quiz_updates.csv` file is used by the skill profile fusion module to:
1. Load quiz-verified proficiency scores
2. Combine with transcript-based scores
3. Create final skill profile with both evidence sources

**Workflow:**
```
Transcript → Skill Scores (prior belief)
    +
Quiz → Skill Proficiency (verified evidence)
    ↓
Fused Skill Profile (final validated scores)
```

---

## ⚠️ Manual Setup Notes

**None required!** Quiz scoring works automatically.

**File Management:**
- Files are created automatically in `output/` directory
- Old results are preserved (appended for detailed, replaced for updates)
- Files can be cleared manually if needed for testing

---

## 🎯 Summary

**Status**: ✅ **COMPLETE**

All requirements for Step 7 (Quiz Scoring) are implemented:
- ✅ Accepts question_id, selected_option, response_time_seconds
- ✅ Calculates overall accuracy
- ✅ Calculates per-skill accuracy
- ✅ Generates detailed results per question
- ✅ Saves to quiz_results_scored.csv
- ✅ Saves to skill_quiz_updates.csv
- ✅ Returns all results to frontend for display

**Integration**: Results are saved and ready for skill profile fusion, enabling the system to combine transcript-based scores with quiz-verified proficiency.

