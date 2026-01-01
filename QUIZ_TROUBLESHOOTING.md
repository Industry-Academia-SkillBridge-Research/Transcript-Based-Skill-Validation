# Quiz Generation Troubleshooting Guide

## Issue: Quiz Loading Page (Empty/Stuck)

If clicking "Generate Quiz" shows a loading page that never finishes, check the following:

### 1. Check Browser Console (F12)

Open Developer Tools (F12) and check the Console tab for errors:

**Common Errors:**
- `Failed to fetch` → Backend is not running
- `404 Not Found` → API endpoint issue
- `500 Internal Server Error` → Backend error (check backend terminal)
- `CORS error` → Backend CORS configuration issue

### 2. Check Backend Terminal

Look for error messages in the backend terminal where `uvicorn` is running:

**Common Backend Errors:**
- `GEMINI_API_KEY not found` → API key not set
- `skill_corpus.csv not found` → Corpus file missing
- `No questions generated` → Corpus empty or skill not found
- `ModuleNotFoundError` → Missing Python packages

### 3. Verify Prerequisites

#### Check API Key is Set

**Windows PowerShell:**
```powershell
echo $env:GEMINI_API_KEY
```

If empty, set it:
```powershell
[System.Environment]::SetEnvironmentVariable("GEMINI_API_KEY", "YOUR_KEY", "User")
# RESTART terminal after this!
```

**Test API Key:**
```powershell
cd backend
python test_gemini_key.py
```

Should output: `OK`

#### Check Skill Corpus Exists

```powershell
Test-Path backend\content\skill_corpus.csv
```

Should return: `True`

If `False`, build corpus:
```powershell
cd backend
python src/build_skill_corpus_chunks.py
```

#### Check Corpus Has Content

```powershell
Get-Content backend\content\skill_corpus.csv | Measure-Object -Line
```

Should show at least 10+ lines (including header).

### 4. Check Network Tab

1. Open Developer Tools (F12)
2. Go to **Network** tab
3. Click "Generate Quiz"
4. Look for the request to `/students/{id}/prepare-quiz`
5. Check:
   - **Status Code**: Should be `200`, not `400`, `404`, or `500`
   - **Response**: Click on the request, go to "Response" tab, see if there's an error message

### 5. Common Fixes

#### Fix 1: Backend Not Running

**Problem:** `Failed to fetch` error

**Solution:**
```powershell
cd backend
.\.venv\Scripts\Activate.ps1
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

#### Fix 2: API Key Not Set

**Problem:** Backend shows `GEMINI_API_KEY not found`

**Solution:**
1. Set environment variable (see Step 3 above)
2. **Restart terminal**
3. Restart backend server

#### Fix 3: Corpus Missing

**Problem:** `skill_corpus.csv not found` or `No questions generated`

**Solution:**
```powershell
cd backend
python src/build_skill_corpus_chunks.py
```

Or manually add chunks:
```powershell
python src/add_corpus_chunks.py
```

#### Fix 4: Selected Skill Not in Corpus

**Problem:** Quiz generates but returns empty questions

**Solution:**
1. Check if your selected skill exists in corpus:
   ```powershell
   Import-Csv backend\content\skill_corpus.csv | Select-Object Skill -Unique
   ```

2. If skill is missing, add it:
   ```powershell
   python src/add_corpus_chunks.py
   ```

#### Fix 5: CORS Error

**Problem:** Browser shows CORS error

**Solution:**
Check `backend/src/api/main.py` has CORS middleware:
```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
```

#### Fix 6: Frontend Stuck in Loading State

**Problem:** Loading spinner never stops

**Solution:**
1. Check browser console for JavaScript errors
2. Check Network tab - is the request pending or failed?
3. Try hard refresh: `Ctrl+Shift+R` or `Ctrl+F5`
4. Clear browser cache

### 6. Debug Steps

#### Step 1: Test Backend Directly

Open in browser: http://127.0.0.1:8000/docs

1. Expand `POST /students/{student_id}/prepare-quiz`
2. Click "Try it out"
3. Enter:
   - `student_id`: Your student ID (e.g., "IT21013928")
   - `selected_skills`: ["Python"] (or your skill)
   - `num_questions_per_skill`: 3
   - `difficulty`: "mixed"
4. Click "Execute"
5. Check response - if error, error message will show here

#### Step 2: Check Backend Logs

Look at the terminal where backend is running for detailed error messages.

#### Step 3: Check Generated Quiz Files

After clicking "Generate Quiz", check if files were created:

```powershell
Get-ChildItem backend\output\quizzes\*.json | Sort-Object LastWriteTime -Descending | Select-Object -First 2
```

Should show:
- `{quiz_id}_questions.json`
- `{quiz_id}_meta.json`

If these files exist, quiz generation worked, but frontend might have an issue loading them.

### 7. Quick Checklist

Before reporting an issue, verify:

- [ ] Backend is running (http://127.0.0.1:8000/docs accessible)
- [ ] Frontend is running (http://localhost:5173 accessible)
- [ ] API key is set (`echo $env:GEMINI_API_KEY` shows your key)
- [ ] API key is valid (`python backend/test_gemini_key.py` shows "OK")
- [ ] Corpus exists (`Test-Path backend\content\skill_corpus.csv` returns True)
- [ ] Corpus has content (at least 10+ lines)
- [ ] Student ID is entered/correct
- [ ] Skills are selected (1-5 skills)
- [ ] Browser console shows no errors
- [ ] Network tab shows successful API calls

### 8. Still Not Working?

If all above checks pass but quiz still doesn't work:

1. **Share error messages:**
   - Browser console errors
   - Backend terminal errors
   - Network tab response

2. **Test API directly:**
   - Use Postman or http://127.0.0.1:8000/docs
   - Try the `/prepare-quiz` endpoint manually

3. **Check file permissions:**
   - Make sure backend can write to `output/quizzes/` directory

4. **Restart everything:**
   - Stop backend (Ctrl+C)
   - Stop frontend (Ctrl+C)
   - Close terminals
   - Restart backend
   - Restart frontend

