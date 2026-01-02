# Skill Profile Dashboard Implementation

## Overview

After quiz completion, students can now view a comprehensive **Skill Profile Dashboard** that shows:
- **Transcript Score**: Skill level inferred from academic transcript (courses and grades)
- **Quiz Score**: Performance on the validation quiz for each skill
- **Final Score**: Combined score using dynamic weighting (transcript + quiz validation)
- **Skill Level**: Verified skill level after quiz validation (Beginner/Developing/Proficient/Advanced)

---

## What Was Implemented

### 1. Skill Profile Dashboard Component

**File**: `frontend-react/src/components/SkillProfileDashboard.jsx`

**Features**:
- Displays quizzed skills in a detailed table
- Shows summary statistics (Total Skills, Avg Transcript Score, Avg Quiz Score, Avg Final Score)
- Color-coded scores and skill levels
- Filters to show only skills that were quizzed (if `selectedSkills` provided)
- Legend explaining the different score types

**Data Displayed**:
- Skill name
- Skill level badge (color-coded: Beginner/Developing/Proficient/Advanced)
- Transcript score (percentage)
- Quiz score (percentage, or "Not quizzed" if not available)
- Final score (percentage)
- Number of quiz questions answered

### 2. Navigation Integration

**Updated Files**:
- `frontend-react/src/App.jsx`: Added `showSkillDashboard` state and navigation logic
- `frontend-react/src/components/QuizResultPage.jsx`: Added "View Skill Profile Dashboard" button

**User Flow**:
1. Student completes quiz
2. Student sees quiz results page
3. Student clicks "View Skill Profile Dashboard" button
4. Dashboard displays with all quizzed skills and their scores

### 3. Backend Data Support

The backend already provides all required fields through `/students/{student_id}/skills` endpoint:
- `ScoreNormalized`: Transcript-based score (0-1)
- `QuizProficiency`: Quiz performance (0-1, null if not quizzed)
- `FinalScore`: Fused score (0-1)
- `FinalSkillLevel`: Verified skill level
- `NumQuestions`: Number of quiz questions answered

The endpoint prioritizes `skill_profiles_with_quiz.csv` (fused profile) which is automatically created after quiz submission.

---

## Usage

### For Students

1. Complete a quiz by selecting skills and answering questions
2. After submitting the quiz, you'll see the Quiz Results page
3. Click **"View Skill Profile Dashboard"** button
4. Review your verified skill profile:
   - See how your transcript scores compare to quiz performance
   - Understand how the final score is calculated (weighted combination)
   - Identify skills that improved or need work based on quiz validation

### For Developers

The dashboard component can be used independently:

```jsx
<SkillProfileDashboard
  studentId="IT21013928"
  selectedSkills={["Python", "SQL"]} // Optional: filter to specific skills
  onBack={() => navigateBack()}
/>
```

If `selectedSkills` is provided, only those skills will be shown. If empty, all skills from the student's profile will be displayed.

---

## Data Fields Reference

### Score Normalization
All scores are displayed as percentages (0-100%) but stored as decimals (0-1) in the backend.

### Skill Levels
- **Beginner** (0-25%): Basic understanding
- **Developing** (25-50%): Intermediate knowledge
- **Proficient** (50-75%): Advanced knowledge
- **Advanced** (75-100%): Expert level

### Score Calculation
The Final Score is calculated using dynamic weights:
- **Quiz Weight**: `min(0.7, NumQuestions / 10.0)` - Increases with more questions
- **Transcript Weight**: `1 - Quiz Weight`
- **Final Score**: `(TranscriptWeight × TranscriptScore) + (QuizWeight × QuizScore)`

This means:
- 1-3 questions: Quiz has small influence (10-30%)
- 7-10+ questions: Quiz dominates (up to 70%)
- More questions = more reliable quiz score = higher weight

---

## Files Modified

1. ✅ `frontend-react/src/components/SkillProfileDashboard.jsx` - New component
2. ✅ `frontend-react/src/components/QuizResultPage.jsx` - Added dashboard button
3. ✅ `frontend-react/src/App.jsx` - Added dashboard navigation
4. ✅ `frontend-react/src/components/QuizPage.jsx` - Enhanced to collect violations

---

## Next Steps (Optional Enhancements)

1. **Filtering**: Add filters by skill level, score range, or quizzed/unquizzed
2. **Sorting**: Allow sorting by any column
3. **Export**: Add ability to export dashboard as PDF/CSV
4. **Charts**: Visualize score trends over time (if multiple quizzes taken)
5. **Comparison**: Compare transcript vs quiz vs final scores with visual charts

---

## Notes

- The dashboard automatically loads the latest skill profile after quiz completion
- Scores are fetched from the backend API (`/students/{student_id}/skills`)
- The dashboard shows "N/A" for skills that haven't been quizzed yet
- All scores are displayed with 1 decimal place for clarity

