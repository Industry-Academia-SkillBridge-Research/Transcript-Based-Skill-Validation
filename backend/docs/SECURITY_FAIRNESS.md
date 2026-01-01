# Security and Fairness Features

## Overview

The quiz system implements light, realistic security and fairness measures suitable for a research prototype:

1. ✅ **Hide correct answers until submission** - Answers never sent to frontend
2. ✅ **Time limit enforcement** - `time_limit_seconds` in quiz metadata
3. ✅ **Shuffle options per question** - Frontend shuffles A/B/C/D options
4. ✅ **Shuffle question order** - Frontend randomizes question sequence

## Implementation Details

### 1. Hide Correct Answers Until Submission

**Backend:**
- Questions stored with `CorrectOption` field in `{quiz_id}_questions.json`
- Answer key stored in `{quiz_id}_meta.json`
- Questions sent to frontend have NO `CorrectOption` field

**Frontend:**
- Never receives answer information
- Can only see correct answers after submission via results page

**Security:** ✅ Answers stored server-side only, never exposed to client

### 2. Time Limit

**Backend:**
- Calculated automatically: `max(10, num_questions * 2)` minutes
- Stored in metadata as `time_limit_minutes` and `time_limit_seconds`
- Returned to frontend in quiz response

**Frontend:**
- Receives `time_limit_seconds` from backend
- Can implement countdown timer (optional, not enforced server-side)

**Example:**
```json
{
  "time_limit_minutes": 12,
  "time_limit_seconds": 720
}
```

### 3. Shuffle Options Per Question

**Implementation:**
- Fisher-Yates shuffle algorithm for fair randomization
- Each question's options (A, B, C, D) are shuffled independently
- Mapping stored to convert shuffled selections back to original A/B/C/D

**Process:**
1. Question received: `{OptionA: "Python", OptionB: "Java", OptionC: "C++", OptionD: "Ruby"}`
2. Frontend shuffles: `[C++, Java, Python, Ruby]`
3. Displayed as: `A: C++, B: Java, C: Python, D: Ruby`
4. User selects "C" (which is actually "Python")
5. On submit: Map "C" back to original "A" before sending to backend

**Code:**
- `frontend-react/src/utils/shuffle.js` - Shuffle utilities
- `frontend-react/src/components/QuizPage.jsx` - Shuffling logic

**Security:** ✅ Prevents memorizing answer positions (A=correct, etc.)

### 4. Shuffle Question Order

**Implementation:**
- Questions array shuffled using Fisher-Yates algorithm
- Each quiz instance has different question order
- Order is randomized client-side when quiz loads

**Process:**
1. Backend sends questions: `[Q1, Q2, Q3, Q4, Q5]`
2. Frontend shuffles: `[Q3, Q1, Q5, Q2, Q4]`
3. User sees shuffled order
4. On submit: Map back to original question IDs (not order)

**Code:**
- `shuffleArray()` function in `frontend-react/src/utils/shuffle.js`
- Applied to questions array in `QuizPage` component

**Security:** ✅ Prevents memorizing question sequences

## Fairness Guarantees

### What This Provides

1. **Answer Protection:** Answers never exposed to client before submission
2. **Option Randomization:** Same question won't always have answer in same position
3. **Question Randomization:** Different quiz instances have different orders
4. **Time Awareness:** Frontend knows time limit (enforcement is optional)

### What This Does NOT Provide

1. **Exam Lockdown:** No browser lockdown or screen recording
2. **Server-Side Timer Enforcement:** Backend doesn't reject late submissions (optional)
3. **Tab Switching Prevention:** Users can still switch tabs/windows
4. **Copy Prevention:** Users can copy questions/answers

**Rationale:** For a research prototype, these measures provide sufficient fairness without complex lockdown mechanisms. They prevent simple cheating (memorizing positions) while maintaining usability.

## Technical Implementation

### Shuffling Algorithm

```javascript
// Fisher-Yates shuffle (fair randomization)
function shuffleArray(array) {
  const shuffled = [...array];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}
```

### Option Mapping

```javascript
// When shuffling options for a question
{
  shuffledOptions: [
    {original: 'C', text: 'Option C text'},
    {original: 'A', text: 'Option A text'},
    {original: 'D', text: 'Option D text'},
    {original: 'B', text: 'Option B text'}
  ],
  optionMapping: {
    'A': 'C',  // Displayed A -> Original C
    'B': 'A',  // Displayed B -> Original A
    'C': 'D',  // Displayed C -> Original D
    'D': 'B'   // Displayed D -> Original B
  }
}
```

### Submission Mapping

```javascript
// When user selects "B" (displayed option)
// Map to original "A" before sending to backend
const originalOption = optionMapping['B']; // Returns 'A'
submit({ question_id: 123, selected_option: 'A' }); // Send original
```

## File Locations

- **Shuffle utilities:** `frontend-react/src/utils/shuffle.js`
- **Quiz component:** `frontend-react/src/components/QuizPage.jsx`
- **Backend metadata:** `backend/src/api/main.py` (prepare_quiz endpoint)

## Testing

To verify shuffling works:

1. **Option Shuffling:**
   - Generate quiz with same question multiple times
   - Verify options appear in different positions each time

2. **Question Order:**
   - Generate quiz multiple times
   - Verify questions appear in different order each time

3. **Answer Mapping:**
   - Answer questions with shuffled options
   - Verify backend receives correct original option letters (A/B/C/D)
   - Verify scoring is correct

## Future Enhancements (Optional)

If stricter security is needed:

1. **Server-Side Timer Enforcement:**
   - Store quiz start time
   - Reject submissions after time_limit_seconds

2. **Question Display Time Tracking:**
   - Track time per question
   - Detect suspicious patterns (all answered instantly)

3. **Client-Side Timer with Warning:**
   - Countdown display
   - Auto-submit when time expires

4. **Session Management:**
   - Prevent multiple simultaneous quiz sessions
   - Track quiz attempt history

