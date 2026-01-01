# Quiz Attempt Controls - Implementation Status

## ✅ REQUIREMENT 6: Quiz Attempt Controls (Advanced, but Realistic)

**Goal**: Implement practical security measures for quiz attempts without requiring full "lockdown browser" functionality.

**Approach**: Lightweight, realistic controls suitable for research prototype with strong demo value.

---

## Status: ✅ **FULLY IMPLEMENTED**

### ✅ Backend: Session Token & Time Limit Enforcement
**Status**: ✅ **COMPLETE**

#### 1. Quiz Session Token
- ✅ **Token Generation**: UUID generated when quiz is created
- ✅ **Token Storage**: Stored in quiz metadata (`session_token` field)
- ✅ **Token Return**: Sent to frontend in quiz response
- ✅ **Token Validation**: Validated on submission (if provided)

**Files:**
- `backend/src/api/main.py` - `prepare_quiz` endpoint

**Implementation:**
```python
# Generate quiz session token
session_token = str(uuid.uuid4())
start_time = datetime.now()

quiz_meta = {
    ...
    "session_token": session_token,
    "session_start_time": start_time.isoformat(),
}
```

#### 2. Start Time Storage
- ✅ **Start Time**: Recorded when quiz is generated
- ✅ **Storage**: Stored in quiz metadata (`session_start_time` field)
- ✅ **Return**: Sent to frontend in quiz response

#### 3. Time Limit Enforcement
- ✅ **Time Validation**: Checks elapsed time vs. time limit on submission
- ✅ **Rejection**: Returns HTTP 400 if time limit exceeded
- ✅ **Error Message**: Clear message showing time limit and elapsed time

**Implementation:**
```python
if session_start_time_str and time_limit_seconds:
    elapsed_seconds = (current_time - session_start_time).total_seconds()
    
    if elapsed_seconds > time_limit_seconds:
        raise HTTPException(
            status_code=400,
            detail=f"Quiz time limit exceeded. Time limit: {time_limit_seconds}s, Elapsed: {elapsed_seconds:.1f}s"
        )
```

**Files:**
- `backend/src/api/main.py` - `submit_quiz` endpoint

---

### ✅ Frontend: Security Controls
**Status**: ✅ **COMPLETE**

#### 1. Fullscreen Mode Request
- ✅ **Auto-Request**: Automatically requests fullscreen when quiz starts
- ✅ **Cross-Browser**: Supports standard, webkit, and ms prefixes
- ✅ **Exit Detection**: Logs violation if user exits fullscreen
- ✅ **Graceful Fallback**: Continues if fullscreen denied/unavailable

**Files:**
- `frontend-react/src/utils/quizSecurity.js` - `requestFullscreen()` function

#### 2. Tab Switch Detection (Visibility Change)
- ✅ **Detection**: Monitors `document.visibilitychange` event
- ✅ **Warning**: Shows alert when user returns to tab
- ✅ **Logging**: Logs violation on first tab switch
- ✅ **Repeated Detection**: Tracks multiple tab switches

**Implementation:**
```javascript
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    logViolation("tab_switch", "User switched to another tab/window");
    // Show warning when user returns
    window.addEventListener("focus", () => {
      alert("⚠️ Warning: Tab switch detected...");
    }, { once: true });
  }
});
```

**Files:**
- `frontend-react/src/utils/quizSecurity.js` - `setupVisibilityListener()` function

#### 3. Copy/Paste Prevention
- ✅ **Copy Prevention**: Blocks `copy` event
- ✅ **Paste Prevention**: Blocks `paste` event
- ✅ **Cut Prevention**: Blocks `cut` event
- ✅ **Text Selection**: Prevents text selection (allows radio buttons)
- ✅ **Violation Logging**: Logs each attempt

**Files:**
- `frontend-react/src/utils/quizSecurity.js` - `setupCopyPastePrevention()` function

#### 4. Right-Click Prevention
- ✅ **Context Menu Block**: Prevents `contextmenu` event
- ✅ **Violation Logging**: Logs each right-click attempt

**Files:**
- `frontend-react/src/utils/quizSecurity.js` - `setupRightClickPrevention()` function

#### 5. Keyboard Shortcuts Prevention
- ✅ **Ctrl+C/V/X/A**: Blocks copy/paste/cut/select all
- ✅ **F12**: Blocks Developer Tools
- ✅ **Ctrl+Shift+I**: Blocks Developer Tools shortcut
- ✅ **Ctrl+Shift+J**: Blocks Console shortcut
- ✅ **Violation Logging**: Logs each attempt

**Files:**
- `frontend-react/src/utils/quizSecurity.js` - `setupKeyboardShortcuts()` function

#### 6. Violation Logging
- ✅ **Logging**: All violations logged with timestamp, type, message
- ✅ **Metadata**: Includes URL, user agent
- ✅ **Collection**: Violations collected in array
- ✅ **Submission**: Violations sent to backend on quiz submission
- ✅ **Backend Storage**: Saved to `{quiz_id}_violations.json`

**Violation Types:**
- `fullscreen_denied` - User denied fullscreen request
- `fullscreen_exit` - User exited fullscreen
- `tab_switch` - User switched tabs
- `tab_switch_repeated` - Multiple tab switches
- `copy_attempt` - Copy attempt
- `paste_attempt` - Paste attempt
- `cut_attempt` - Cut attempt
- `right_click` - Right-click attempt
- `keyboard_shortcut_*` - Keyboard shortcut attempts
- `f12_pressed` - F12 pressed
- `devtools_shortcut` - Developer Tools shortcut
- `console_shortcut` - Console shortcut

**Files:**
- `frontend-react/src/utils/quizSecurity.js` - `logViolation()`, `getViolations()` functions
- `backend/src/api/main.py` - Violation storage in `submit_quiz` endpoint

---

## 📊 Implementation Details

### Backend Session Management

**Quiz Creation:**
```python
session_token = str(uuid.uuid4())
start_time = datetime.now()

quiz_meta = {
    "session_token": session_token,
    "session_start_time": start_time.isoformat(),
    "time_limit_seconds": time_limit_seconds,
    ...
}
```

**Quiz Submission:**
```python
# Validate session token
if payload.session_token:
    expected_token = meta.get("session_token")
    if payload.session_token != expected_token:
        raise HTTPException(status_code=403, detail="Invalid session token")

# Validate time limit
elapsed_seconds = (current_time - session_start_time).total_seconds()
if elapsed_seconds > time_limit_seconds:
    raise HTTPException(status_code=400, detail="Quiz time limit exceeded")

# Log violations
if payload.violations:
    # Save to {quiz_id}_violations.json
```

### Frontend Security Initialization

**Usage in QuizPage:**
```javascript
import { initQuizSecurity, getViolations } from "../utils/quizSecurity";

useEffect(() => {
  if (quizGenerated) {
    const cleanup = initQuizSecurity((violation) => {
      // Handle violation (optional callback)
      console.warn("Security violation:", violation);
    });
    
    return cleanup; // Cleanup on unmount
  }
}, [quizGenerated]);

// On submit
const violations = getViolations();
await submitQuiz(studentId, responses, quizId, sessionToken, violations);
```

---

## 🔒 Security Level

**What This Provides:**
- ✅ **Time Limit Enforcement**: Server-side validation prevents late submissions
- ✅ **Session Validation**: Token ensures quiz submission matches original session
- ✅ **Violation Tracking**: All security violations logged and stored
- ✅ **User Deterrent**: Visible warnings and blocked actions discourage cheating
- ✅ **Research Value**: Violation logs provide data for analysis

**Limitations (Realistic for Research Prototype):**
- ⚠️ **Not Perfect Security**: Determined users can bypass client-side controls
- ⚠️ **Browser-Dependent**: Some controls depend on browser support
- ⚠️ **No Network Blocking**: Cannot prevent external resources
- ⚠️ **No Screen Recording Block**: Cannot prevent screen recording

**Why This Is Acceptable:**
- ✅ **Research Prototype**: Not production exam system
- ✅ **Strong Demo Value**: Shows security awareness
- ✅ **Realistic Approach**: Acknowledges limitations
- ✅ **Violation Data**: Provides research data on behavior

---

## 📝 Violation Log Format

**Frontend Violation:**
```json
{
  "type": "tab_switch",
  "message": "User switched to another tab/window",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "url": "http://localhost:5173/quiz",
  "userAgent": "Mozilla/5.0..."
}
```

**Backend Storage (`{quiz_id}_violations.json`):**
```json
{
  "quiz_id": "550e8400-e29b-41d4-a716-446655440000",
  "student_id": "IT21013928",
  "violations": [...],
  "timestamp": "2024-01-15T10:35:00.000Z",
  "violation_logs": [
    {
      "quiz_id": "...",
      "student_id": "...",
      "violations": [...],
      "timestamp": "..."
    }
  ]
}
```

---

## ✅ Verification Checklist

**Backend:**
- [x] Session token generated on quiz creation
- [x] Start time stored in metadata
- [x] Session token returned to frontend
- [x] Time limit validation on submission
- [x] Submission rejected if time exceeded
- [x] Session token validation (if provided)
- [x] Violations logged to file

**Frontend:**
- [x] Fullscreen mode requested
- [x] Tab switch detection
- [x] Copy/paste prevention
- [x] Right-click prevention
- [x] Keyboard shortcuts blocked
- [x] Violations logged
- [x] Violations sent to backend on submit
- [x] Cleanup on component unmount

---

## 🎯 Summary

**Status**: ✅ **COMPLETE**

All requirements for Step 6 (Quiz Attempt Controls) are implemented:
- ✅ Backend: Session token, start time storage, time limit enforcement
- ✅ Frontend: Fullscreen, tab switch detection, copy/paste/right-click prevention
- ✅ Violation logging and storage
- ✅ Realistic security measures suitable for research prototype

**Security Level**: Practical and realistic for a research prototype, with strong demo value and research data collection capabilities.

