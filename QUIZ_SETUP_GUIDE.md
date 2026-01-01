# Quiz Setup Guide - Manual Steps Required

## 📋 Overview

To use the quiz generation feature, you need to:

1. ✅ Set up **Google Gemini API Key** (REQUIRED for quiz generation)
2. ✅ Build **Skill Corpus** (REQUIRED - knowledge base for questions)
3. ✅ Optional: Configure retrieval method (TF-IDF or Embeddings)

---

## 🔑 Step 1: Get Google Gemini API Key

### 1.1 Get API Key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Sign in with your Google account
3. Click **"Get API Key"** or **"Create API Key"**
4. Copy your API key (starts with `AIza...`)

### 1.2 Set Environment Variable

#### Windows (PowerShell)

**Option A: Set for Current Session** (temporary, lost when terminal closes)
```powershell
$env:GEMINI_API_KEY = "YOUR_API_KEY_HERE"
```

**Option B: Set Permanently** (recommended)
```powershell
[System.Environment]::SetEnvironmentVariable("GEMINI_API_KEY", "YOUR_API_KEY_HERE", "User")
```

After setting permanently, **restart your terminal/PowerShell**.

#### Windows (Command Prompt)
```cmd
setx GEMINI_API_KEY "YOUR_API_KEY_HERE"
```

After running this, **close and reopen** your terminal.

#### Linux/Mac
```bash
export GEMINI_API_KEY="YOUR_API_KEY_HERE"
```

Or add to `~/.bashrc` or `~/.zshrc`:
```bash
echo 'export GEMINI_API_KEY="YOUR_API_KEY_HERE"' >> ~/.bashrc
source ~/.bashrc
```

### 1.3 Verify API Key

Run the test script:
```powershell
cd backend
python test_gemini_key.py
```

Expected output: `OK`

If you see an error, check:
- ✅ API key is set correctly
- ✅ API key is valid (not expired)
- ✅ You have internet connection
- ✅ You restarted your terminal after setting the key

---

## 📚 Step 2: Build Skill Corpus

The quiz system needs a **knowledge corpus** to generate questions from. This is a CSV file with skill knowledge chunks.

### 2.1 Check if Corpus Exists

Check if `backend/content/skill_corpus.csv` exists:
```powershell
Test-Path backend\content\skill_corpus.csv
```

### 2.2 Build Corpus from Knowledge Base

If you have knowledge base files (text files with skill content):

1. **Place knowledge files** in `backend/knowledge_base/`:
   ```
   backend/knowledge_base/
   ├── SQL.txt
   ├── Python.txt
   ├── Database_Design.txt
   └── ...
   ```

2. **Run the corpus builder**:
   ```powershell
   cd backend
   python src\build_skill_corpus_chunks.py
   ```

   This will:
   - Read all `.txt` files from `knowledge_base/`
   - Split into chunks (150-300 words each)
   - Create `content/skill_corpus.csv`

3. **Verify corpus was created**:
   ```powershell
   Get-Content backend\content\skill_corpus.csv | Select-Object -First 5
   ```

   Expected format:
   ```csv
   Skill,ChunkID,Text,Source
   SQL,sql_chunk_001,"SQL is a database language...",IT1010 - Database Systems
   ```

### 2.3 Add Skills Manually (Alternative)

If you don't have knowledge base files, you can add skills manually:

```powershell
cd backend
python src\add_corpus_chunks.py
```

Or use the CSV format directly - see `backend/docs/BUILD_SKILL_CORPUS.md` for details.

### 2.4 Required Corpus Structure

The corpus CSV must have these columns:
- **Skill**: Skill name (e.g., "SQL", "Python")
- **ChunkID**: Unique chunk identifier (e.g., "sql_chunk_001")
- **Text**: Chunk content (150-300 words recommended)
- **Source**: Optional source reference (e.g., "IT1010 - Database Systems")

**Important**: Each skill should have **multiple chunks** (at least 3-5) for good question generation.

---

## ⚙️ Step 3: Configure Retrieval Method (Optional)

By default, the system uses **TF-IDF** for retrieval. To use **embeddings** (better results):

### 3.1 Install Embeddings Dependencies

```powershell
cd backend
pip install sentence-transformers faiss-cpu
```

### 3.2 Set Environment Variable

```powershell
$env:RAG_RETRIEVAL_METHOD = "embeddings"
```

Or use TF-IDF (default):
```powershell
$env:RAG_RETRIEVAL_METHOD = "tfidf"
```

---

## ✅ Step 4: Verify Everything Works

### 4.1 Start Backend

```powershell
cd backend
.\run-backend.ps1
```

### 4.2 Start Frontend

In a new terminal:
```powershell
cd frontend-react
npm run dev
```

### 4.3 Test Quiz Generation

1. Open frontend: http://localhost:5173
2. Upload a transcript (if you have one)
3. Select skills from the Skills page
4. Click "Generate Quiz"
5. Check if questions are generated

**If you see errors:**
- Check backend terminal for error messages
- Verify API key is set: `echo $env:GEMINI_API_KEY` (PowerShell)
- Verify corpus exists: `Test-Path backend\content\skill_corpus.csv`
- Check corpus has content: `Get-Content backend\content\skill_corpus.csv | Measure-Object -Line`

---

## 📝 Quick Checklist

Before generating quizzes, verify:

- [ ] ✅ Gemini API key is set (`GEMINI_API_KEY` environment variable)
- [ ] ✅ API key is valid (test with `python test_gemini_key.py`)
- [ ] ✅ Skill corpus exists (`backend/content/skill_corpus.csv`)
- [ ] ✅ Corpus has chunks for the skills you want to quiz
- [ ] ✅ Backend dependencies installed (`pip install -r requirements.txt`)
- [ ] ✅ Backend is running (`run-backend.ps1`)
- [ ] ✅ Frontend is running (`npm run dev`)

---

## 🔍 Troubleshooting

### "API key not found" error

**Solution:**
1. Verify key is set: `echo $env:GEMINI_API_KEY` (PowerShell)
2. If empty, set it again (see Step 1.2)
3. **Restart terminal/PowerShell** after setting
4. Restart backend server

### "No questions generated" error

**Possible causes:**

1. **Corpus missing or empty:**
   ```powershell
   Test-Path backend\content\skill_corpus.csv
   Get-Content backend\content\skill_corpus.csv | Measure-Object -Line
   ```
   Solution: Build corpus (Step 2)

2. **No chunks for selected skill:**
   ```powershell
   Get-Content backend\content\skill_corpus.csv | Select-String "SQL"
   ```
   Solution: Add chunks for that skill

3. **API key invalid:**
   Solution: Verify with `python test_gemini_key.py`

4. **Fallback to question bank:**
   - System will use `question_bank.csv` if Gemini fails
   - Check if question bank has questions for selected skills

### "Module not found" errors

**Solution:**
```powershell
cd backend
pip install -r requirements.txt
```

### Questions are too similar/repetitive

**This is handled automatically!** The system:
- ✅ Instructs Gemini to mix question types
- ✅ Detects and removes duplicates automatically
- ✅ Ensures diversity in question formats

If you still see similar questions, check:
- Corpus has diverse content (not repetitive)
- Corpus has enough chunks per skill (3-5+ chunks)

---

## 📖 Additional Resources

- **Building Corpus**: `backend/docs/BUILD_SKILL_CORPUS.md`
- **Corpus Quick Start**: `backend/CORPUS_QUICKSTART.md`
- **RAG Retrieval**: `backend/docs/RAG_RETRIEVAL.md`
- **Question Diversity**: `backend/docs/QUESTION_DIVERSITY.md`
- **Quiz Storage**: `backend/docs/QUIZ_STORAGE.md`

---

## 🎯 Next Steps

Once everything is set up:

1. **Test with one skill** first (e.g., "SQL")
2. **Generate a quiz** and verify questions are diverse
3. **Add more skills** to your corpus as needed
4. **Use question bank** as fallback for skills without corpus

**You're ready to generate quizzes!** 🎉

