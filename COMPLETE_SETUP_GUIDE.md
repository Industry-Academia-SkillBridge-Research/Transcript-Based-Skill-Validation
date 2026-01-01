# 🚀 Complete Setup Guide - Everything You Need

This is your **one-stop guide** to set up and run the entire Transcript-Based Skill Validation system.

---

## 📋 Prerequisites

Before starting, ensure you have:

- ✅ **Python 3.8+** installed (check: `python --version`)
- ✅ **Node.js 16+ and npm** installed (check: `node --version` and `npm --version`)
- ✅ **PowerShell** (Windows) or **Bash** (Linux/Mac)
- ✅ **Git** (if cloning from repository)

---

## 🔑 Step 1: Set Up Google Gemini API Key (REQUIRED for Quiz Generation)

### 1.1 Get Your API Key

1. Go to **[Google AI Studio](https://aistudio.google.com/apikey)**
2. Sign in with your Google account
3. Click **"Get API Key"** or **"Create API Key"**
4. Copy your API key (starts with `AIza...`)

### 1.2 Set Environment Variable

#### Windows (PowerShell) - **RECOMMENDED (Permanent)**

```powershell
[System.Environment]::SetEnvironmentVariable("GEMINI_API_KEY", "YOUR_API_KEY_HERE", "User")
```

**⚠️ IMPORTANT:** After running this, **close and restart your terminal/PowerShell** for the change to take effect.

#### Windows (PowerShell) - Temporary (Current Session Only)

```powershell
$env:GEMINI_API_KEY = "YOUR_API_KEY_HERE"
```

#### Windows (Command Prompt)

```cmd
setx GEMINI_API_KEY "YOUR_API_KEY_HERE"
```

**⚠️ IMPORTANT:** Close and reopen your terminal after running this.

#### Linux/Mac

```bash
export GEMINI_API_KEY="YOUR_API_KEY_HERE"
```

Or add to `~/.bashrc` or `~/.zshrc` for permanent setup:
```bash
echo 'export GEMINI_API_KEY="YOUR_API_KEY_HERE"' >> ~/.bashrc
source ~/.bashrc
```

### 1.3 Verify API Key is Set

**Windows (PowerShell):**
```powershell
echo $env:GEMINI_API_KEY
```

**Linux/Mac:**
```bash
echo $GEMINI_API_KEY
```

**Test API Key:**
```powershell
cd backend
python test_gemini_key.py
```

You should see "OK" if the key works.

---

## 📦 Step 2: Backend Setup

### 2.1 Navigate to Project Root

```powershell
cd "D:\OneDrive - Sri Lanka Institute of Information Technology\Research\Transcript-Based Skill Validation and Job Alignment"
```

### 2.2 Create and Activate Virtual Environment

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

You should see `(.venv)` in your terminal prompt.

### 2.3 Install Backend Dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

**Expected packages installed:**
- fastapi, uvicorn
- pandas, numpy, scikit-learn
- pdfplumber, pytesseract
- google-genai
- Pillow, python-multipart

### 2.4 Verify Backend Setup

```powershell
python -c "import fastapi; print('FastAPI installed')"
python -c "import google.genai; print('Google Genai installed')"
```

---

## 🎨 Step 3: Frontend Setup

### 3.1 Navigate to Frontend Directory

**Open a NEW terminal window** (keep backend terminal open if you want to run it later)

```powershell
cd "D:\OneDrive - Sri Lanka Institute of Information Technology\Research\Transcript-Based Skill Validation and Job Alignment\frontend-react"
```

### 3.2 Install Frontend Dependencies

```powershell
npm install
```

This will install:
- React 19
- Vite
- Tailwind CSS
- All other dependencies

**First time installation takes 1-2 minutes.**

---

## 📚 Step 4: Build Skill Corpus (REQUIRED for Quiz Generation)

The skill corpus is the knowledge base that Gemini uses to generate quiz questions.

### 4.1 Check if Corpus Exists

```powershell
Test-Path backend\content\skill_corpus.csv
```

### 4.2 Build/Update Corpus

If the file doesn't exist or you want to rebuild it:

```powershell
cd backend
# Make sure virtual environment is activated
.\.venv\Scripts\Activate.ps1

# Build corpus from course materials
python src/build_skill_corpus_chunks.py
```

**Or manually add chunks:**

```powershell
python src/add_corpus_chunks.py
```

### 4.3 Verify Corpus

```powershell
Get-Content backend\content\skill_corpus.csv | Measure-Object -Line
```

You should see at least 10-20+ lines (including header).

---

## 💼 Step 5: Process Job Data (Optional - for Role Matching)

If you have `Job_data.json` and want role matching:

### 5.1 Convert JSON to CSV

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python src/convert_job_json_to_csv.py
```

### 5.2 Process Job Postings

```powershell
python src/job_postings_ingestion.py
```

This creates:
- `output/job_role_skill_templates_dynamic.csv`

---

## 🚀 Step 6: Run the Application

You need **TWO terminal windows** - one for backend, one for frontend.

### 6.1 Terminal 1: Start Backend

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

**Or use the script:**
```powershell
# From project root
.\run-backend.ps1
```

**Backend will be available at:**
- API: http://127.0.0.1:8000
- API Docs: http://127.0.0.1:8000/docs

**You should see:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

### 6.2 Terminal 2: Start Frontend

**Open a NEW terminal window**

```powershell
cd frontend-react
npm run dev
```

**Or use the script:**
```powershell
# From project root
.\run-frontend.ps1
```

**Frontend will be available at:**
- http://localhost:5173 (or next available port)

**You should see:**
```
VITE v7.x.x  ready in xxx ms

➜  Local:   http://localhost:5173/
```

### 6.3 Access the Application

1. Open your browser
2. Go to: **http://localhost:5173**
3. You should see the upload page

---

## ✅ Quick Setup Checklist

Before using the application, verify:

- [ ] ✅ Python 3.8+ installed
- [ ] ✅ Node.js 16+ installed
- [ ] ✅ Gemini API key set (`echo $env:GEMINI_API_KEY` should show your key)
- [ ] ✅ API key tested (`python backend/test_gemini_key.py` shows "OK")
- [ ] ✅ Backend dependencies installed (`pip list` shows fastapi, uvicorn, etc.)
- [ ] ✅ Frontend dependencies installed (`npm list` in frontend-react directory)
- [ ] ✅ Skill corpus exists (`Test-Path backend\content\skill_corpus.csv` returns True)
- [ ] ✅ Backend is running (http://127.0.0.1:8000/docs accessible)
- [ ] ✅ Frontend is running (http://localhost:5173 accessible)

---

## 🎯 Step 7: First Time Usage

### 7.1 Upload Transcript

1. Open http://localhost:5173
2. Click "Choose File" or drag and drop a transcript (PDF or image)
3. Click "Upload and Extract"
4. Wait for processing (OCR may take 30-60 seconds)

### 7.2 View Skills

1. After upload, you'll see the transcript summary
2. Click "View Skills" or "Continue to Dashboard"
3. Skills are automatically inferred from your courses

### 7.3 Generate Quiz

1. Select up to 5 skills (checkboxes)
2. Click "Generate Quiz"
3. Wait for questions to generate (10-30 seconds)
4. Take the quiz

### 7.4 View Role Recommendations

1. After quiz, skills are updated (fused profile)
2. Go to "Role suggestions" section
3. Click "Refresh role matches"
4. View recommended roles with readiness scores

---

## 🔧 Troubleshooting

### Backend Issues

#### Port 8000 Already in Use
```powershell
# Find process using port 8000
netstat -ano | findstr :8000

# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F

# Or use a different port
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8001
```

#### Module Not Found Errors
```powershell
# Make sure virtual environment is activated
.\.venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### Permission Errors (PowerShell)
```powershell
# Run PowerShell as Administrator, then:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### API Key Not Found
```powershell
# Verify key is set
echo $env:GEMINI_API_KEY

# If empty, set it again (see Step 1.2)
[System.Environment]::SetEnvironmentVariable("GEMINI_API_KEY", "YOUR_KEY", "User")

# RESTART terminal/PowerShell after setting
# Then restart backend
```

### Frontend Issues

#### Port 5173 Already in Use
- Vite automatically tries next available port (5174, 5175, etc.)
- Check the terminal for the actual port

#### Node Modules Errors
```powershell
cd frontend-react
# Delete node_modules and reinstall
Remove-Item -Recurse -Force node_modules
Remove-Item package-lock.json
npm install
```

#### Cannot Connect to Backend
- Verify backend is running (check Terminal 1)
- Verify backend URL in frontend code (should be http://127.0.0.1:8000)
- Check browser console for CORS errors

### Quiz Generation Issues

#### "No questions generated" Error

**Check 1: API Key**
```powershell
echo $env:GEMINI_API_KEY
python backend/test_gemini_key.py
```

**Check 2: Skill Corpus**
```powershell
Test-Path backend\content\skill_corpus.csv
Get-Content backend\content\skill_corpus.csv | Select-Object -First 10
```

**Check 3: Corpus Has Chunks for Your Skills**
```powershell
# View corpus skills
Import-Csv backend\content\skill_corpus.csv | Select-Object Skill -Unique
```

If your selected skill is not in the corpus, add chunks for it:
```powershell
python backend/src/add_corpus_chunks.py
```

#### "Invalid skills selected" Error

This means the skill is not in the student's skill profile. Either:
1. Make sure transcript was uploaded and processed
2. Or the skill doesn't exist in `course_skill_mapping.csv`

---

## 📁 Important File Locations

### Configuration Files
- `backend/input/course_skill_mapping.csv` - Course to skill mapping
- `backend/input/Job_data.json` - Raw job postings (optional)
- `backend/content/skill_corpus.csv` - Knowledge base for quiz generation

### Output Files
- `backend/output/transcript_parsed_single.csv` - Parsed transcript
- `backend/output/skill_profile_{student_id}.csv` - Student skill profile
- `backend/output/quizzes/{quiz_id}_questions.json` - Generated quiz questions
- `backend/output/quizzes/{quiz_id}_meta.json` - Quiz metadata
- `backend/output/role_readiness_dynamic.csv` - Role matching results

### Scripts
- `run-backend.ps1` - Run backend server
- `run-frontend.ps1` - Run frontend server
- `setup.ps1` - Install all dependencies

---

## 🔄 Daily Usage

### Start the Application (Every Time)

**Terminal 1 - Backend:**
```powershell
cd backend
.\.venv\Scripts\Activate.ps1
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

**Terminal 2 - Frontend:**
```powershell
cd frontend-react
npm run dev
```

### Stop the Application

Press `CTRL+C` in each terminal to stop the servers.

---

## 🆘 Getting Help

### Common Commands Reference

```powershell
# Check Python version
python --version

# Check Node version
node --version

# Check if virtual environment is activated (should show .venv)
python -c "import sys; print(sys.prefix)"

# List installed Python packages
pip list

# List installed npm packages
cd frontend-react
npm list

# Check if backend is running
Invoke-WebRequest http://127.0.0.1:8000/docs

# Test Gemini API key
cd backend
python test_gemini_key.py

# View recent errors in backend
# Check Terminal 1 output

# View recent errors in frontend
# Check browser console (F12)
```

### Check Application Status

1. **Backend Status:** http://127.0.0.1:8000/docs
2. **Frontend Status:** http://localhost:5173
3. **Backend Logs:** Check Terminal 1
4. **Frontend Logs:** Check Terminal 2 and Browser Console (F12)

---

## 📝 Notes

- **API Key:** Keep your Gemini API key secret. Never commit it to git.
- **Virtual Environment:** Always activate `.venv` before running backend scripts
- **Two Terminals:** You need both backend and frontend running simultaneously
- **First Upload:** First transcript upload takes longer due to OCR processing
- **Quiz Generation:** Takes 10-30 seconds depending on number of questions
- **Corpus Updates:** Rebuild corpus if you add new course materials

---

## ✨ You're All Set!

Once you complete these steps, you can:
1. ✅ Upload transcripts
2. ✅ View inferred skills
3. ✅ Generate and take quizzes
4. ✅ View role recommendations
5. ✅ See skill validation results

**Happy coding! 🚀**

