# Quick Reference Guide - Key Files & Their Purposes

A quick lookup guide for understanding what each file does.

---

## 🎯 **Core Pipeline Files** (Read First)

| File | Purpose | Key Functions |
|------|---------|---------------|
| `backend/src/api/main.py` | **Main API server** - All REST endpoints | All HTTP endpoints, request/response handling |
| `backend/src/transcript_ingestion.py` | **Transcript parsing** - Extract courses from PDF/images | `parse_transcript_text()` - Main parsing logic |
| `backend/src/skill_aggregation_from_parsed.py` | **Skill inference** - Convert courses to skills | `build_skill_profile_from_parsed()` - Skill scoring |
| `backend/src/quiz_generation_rag.py` | **Quiz generation** - Generate MCQ questions | RAG-based question generation |
| `backend/src/quiz_scoring.py` | **Quiz scoring** - Score student answers | Answer key matching, score calculation |
| `backend/src/job_role_model_dynamic.py` | **Job matching** - Match skills to job roles | Role readiness calculation |
| `backend/src/skill_profile_fusion.py` | **Profile fusion** - Combine transcript + quiz scores | Weighted fusion algorithm |

---

## 📚 **Data Processing**

| File | Purpose |
|------|---------|
| `backend/src/transcript_loader.py` | Load transcript CSV files |
| `backend/src/course_skill_mapping.py` | Load course → skill mappings |
| `backend/src/job_postings_ingestion.py` | Process job postings, extract skills |
| `backend/src/convert_job_json_to_csv.py` | Convert JSON job data to CSV |

---

## 🧠 **Skill Processing**

| File | Purpose |
|------|---------|
| `backend/src/skill_aggregation_explainable.py` | Create explainable skill profiles with evidence |
| `backend/src/skill_profile_fusion.py` | Fuse transcript scores + quiz scores |

---

## 📝 **Quiz System**

| File | Purpose |
|------|---------|
| `backend/src/quiz_planner.py` | Plan which skills to quiz, allocate questions |
| `backend/src/rag_retriever.py` | Retrieve skill context from corpus |
| `backend/src/rag_retrieval.py` | RAG retrieval logic, similarity search |
| `backend/src/quiz_generation_gemini.py` | Generate questions using Gemini LLM |
| `backend/src/question_diversity.py` | Check question diversity, detect duplicates |
| `backend/src/quiz_validation.py` | Validate question quality and format |

---

## 🎨 **Frontend Components**

| File | Purpose | Displays |
|------|---------|----------|
| `frontend-react/src/App.jsx` | **Main app** - Orchestrates all components | Main layout, navigation, state management |
| `frontend-react/src/api.js` | **API client** - All backend calls | HTTP request functions |
| `frontend-react/src/components/UploadTranscript.jsx` | File upload UI | Upload form |
| `frontend-react/src/components/TranscriptDetailsPage.jsx` | Transcript display | Student details, courses, grades |
| `frontend-react/src/components/SkillsPage.jsx` | Skills display & selection | Skill list, selection UI (max 5) |
| `frontend-react/src/components/QuizPage.jsx` | Quiz taking interface | Questions, timer, submission |
| `frontend-react/src/components/QuizResultPage.jsx` | Quiz results | Scores, breakdown, navigation |
| `frontend-react/src/components/SkillProfileDashboard.jsx` | Combined profile view | Transcript + quiz scores dashboard |
| `frontend-react/src/components/JobRecommendations.jsx` | Job recommendations | Role matches, missing skills |
| `frontend-react/src/components/XaiPanel.jsx` | Explainability panel | Skill/role evidence |

---

## 🔧 **Utility & Support**

| File | Purpose |
|------|---------|
| `backend/src/run_full_pipeline_for_student.py` | **Full pipeline script** - Run complete workflow |
| `backend/src/build_skill_corpus_chunks.py` | Build skill corpus chunks for RAG |
| `backend/src/add_corpus_chunks.py` | Add chunks to skill corpus |
| `backend/src/make_skill_keys.py` | Generate skill keys/normalization |
| `backend/src/generate_skill_corpus_template.py` | Generate corpus template |
| `backend/src/auto_fill_skill_corpus.py` | Auto-fill skill corpus content |

---

## 📊 **Configuration & Data**

| File/Directory | Purpose |
|----------------|---------|
| `backend/input/course_skill_mapping.csv` | **Course → Skill mappings** (must maintain) |
| `backend/input/Job_data.json` | Raw job posting data |
| `backend/content/skill_corpus.csv` | **Skill knowledge base** for RAG quiz generation |
| `backend/configs/job_role_templates.json` | Role template configurations |
| `backend/output/` | All generated output files |

---

## 📖 **Documentation Files**

| File | What It Covers |
|------|----------------|
| `PROJECT_STATUS_ASSESSMENT.md` | ⭐ **Overall project status** - Start here |
| `PROJECT_UNDERSTANDING_GUIDE.md` | ⭐ **Complete reading order** - How to understand the project |
| `ARCHITECTURE_OVERVIEW.md` | ⭐ **System architecture** - High-level design |
| `RUN_INSTRUCTIONS.md` | How to run the application |
| `TRANSCRIPT_INGESTION_STATUS.md` | Transcript processing details |
| `SKILL_INFERENCE_STATUS.md` | Skill inference methodology |
| `SKILL_FUSION_STATUS.md` | Skill fusion algorithm |
| `RAG_QUIZ_GENERATION_STATUS.md` | Quiz generation system |
| `QUIZ_SCORING_STATUS.md` | Quiz scoring system |
| `ROLE_MATCHING_STATUS.md` | Job matching algorithm |
| `backend/README.md` | Backend architecture overview |

---

## 🔄 **Common Workflows**

### Workflow 1: Upload & Process Transcript
```
User uploads PDF
  → transcript_ingestion.py (parse)
  → course_skill_mapping.py (load mappings)
  → skill_aggregation_from_parsed.py (infer skills)
  → Output: skill_profile_{student_id}.csv
```

### Workflow 2: Generate & Take Quiz
```
User selects skills (≤5)
  → quiz_planner.py (plan quiz)
  → rag_retriever.py (retrieve context)
  → quiz_generation_gemini.py (generate questions)
  → question_diversity.py (validate)
  → User takes quiz
  → quiz_scoring.py (score answers)
  → Output: quiz results
```

### Workflow 3: Update Profile & Match Jobs
```
Quiz scores available
  → skill_profile_fusion.py (fuse scores)
  → job_role_model_dynamic.py (match roles)
  → Output: role_readiness_dynamic.csv
```

### Workflow 4: Full Pipeline
```
backend/src/run_full_pipeline_for_student.py
  → Runs all steps in sequence
  → Pre-quiz: transcript → skills → jobs
  → Post-quiz: quiz → fusion → updated jobs
```

---

## 🎯 **Key API Endpoints**

| Endpoint | Method | Purpose | Key File |
|----------|--------|---------|----------|
| `/upload-transcript` | POST | Upload and parse transcript | `api/main.py` |
| `/students/{id}/skills` | GET | Get student skill profile | `api/main.py` |
| `/students/{id}/roles` | GET | Get job recommendations | `api/main.py` |
| `/prepare-quiz` | POST | Generate quiz questions | `api/main.py` |
| `/submit-quiz` | POST | Submit quiz answers | `api/main.py` |
| `/students/{id}/xai/skills` | GET | Get skill explanations | `api/main.py` |
| `/students/{id}/xai/roles` | GET | Get role explanations | `api/main.py` |

---

## 🔍 **Where to Find Things**

### "Where is transcript parsing logic?"
→ `backend/src/transcript_ingestion.py`

### "Where are skills calculated from courses?"
→ `backend/src/skill_aggregation_from_parsed.py`

### "Where are quiz questions generated?"
→ `backend/src/quiz_generation_rag.py` or `quiz_generation_gemini.py`

### "Where are jobs matched to students?"
→ `backend/src/job_role_model_dynamic.py`

### "Where is the API server?"
→ `backend/src/api/main.py`

### "Where is the frontend main app?"
→ `frontend-react/src/App.jsx`

### "Where is the full pipeline script?"
→ `backend/src/run_full_pipeline_for_student.py`

### "Where are course-skill mappings?"
→ `backend/input/course_skill_mapping.csv`

### "Where is the skill corpus for RAG?"
→ `backend/content/skill_corpus.csv`

---

## 📋 **File Size & Complexity Indicators**

| File Size | Typical Complexity | Example Files |
|-----------|-------------------|---------------|
| Very Large (>1000 lines) | High complexity, many features | `api/main.py` (1700+ lines), `App.jsx` (1000+ lines) |
| Large (500-1000 lines) | Moderate-high complexity | `job_role_model_dynamic.py`, `quiz_generation_rag.py` |
| Medium (200-500 lines) | Moderate complexity | Most processing files |
| Small (<200 lines) | Lower complexity, focused | Utility files, simple components |

---

## 🚨 **Critical Files to Understand**

1. **`backend/src/api/main.py`** - The heart of the API
2. **`backend/src/transcript_ingestion.py`** - How transcripts are parsed
3. **`backend/src/skill_aggregation_from_parsed.py`** - How skills are inferred
4. **`backend/src/quiz_generation_rag.py`** - How quizzes are generated
5. **`backend/src/job_role_model_dynamic.py`** - How jobs are matched
6. **`frontend-react/src/App.jsx`** - Frontend flow and state

---

## 💡 **Quick Tips**

- **Want to understand the overall flow?** → Read `PROJECT_STATUS_ASSESSMENT.md` first
- **Want to trace a specific feature?** → Check the corresponding `*_STATUS.md` file
- **Want to modify API behavior?** → Focus on `api/main.py`
- **Want to change quiz generation?** → Look at `quiz_generation_*.py` files
- **Want to adjust job matching?** → Check `job_role_model_dynamic.py`
- **Want to understand frontend flow?** → Start with `App.jsx`, then trace components

---

## 🔗 **Related Files Groups**

### Transcript Processing Group
- `transcript_loader.py`
- `transcript_ingestion.py`
- `course_skill_mapping.py`

### Skill Processing Group
- `skill_aggregation_from_parsed.py`
- `skill_aggregation_explainable.py`
- `skill_profile_fusion.py`

### Quiz Generation Group
- `quiz_planner.py`
- `rag_retriever.py`
- `rag_retrieval.py`
- `quiz_generation_rag.py`
- `quiz_generation_gemini.py`
- `question_diversity.py`
- `quiz_validation.py`

### Quiz Scoring Group
- `quiz_scoring.py`
- (scoring logic in `api/main.py`)

### Job Matching Group
- `job_postings_ingestion.py`
- `job_role_model_dynamic.py`
- `convert_job_json_to_csv.py`

---

This quick reference should help you navigate the codebase efficiently!

