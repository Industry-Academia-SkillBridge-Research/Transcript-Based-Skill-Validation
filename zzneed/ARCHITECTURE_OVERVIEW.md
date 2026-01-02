# Architecture Overview - Visual Guide

This document provides a high-level architectural overview of the Transcript-Based Skill Validation and Job Alignment system.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND (React)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ Upload   │→ │  Skills  │→ │   Quiz   │→ │   Jobs   │      │
│  │Transcript│  │  Select  │  │   Take   │  │ Recommend│      │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
└───────────────────────────────┬─────────────────────────────────┘
                                │ HTTP/REST API
┌───────────────────────────────▼─────────────────────────────────┐
│                    BACKEND (FastAPI)                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  API Endpoints (main.py)                 │  │
│  │  • /upload-transcript                                    │  │
│  │  • /students/{id}/skills                                 │  │
│  │  • /prepare-quiz                                         │  │
│  │  • /submit-quiz                                          │  │
│  │  • /students/{id}/roles                                  │  │
│  │  • /students/{id}/xai/skills                             │  │
│  │  • /students/{id}/xai/roles                              │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Data Flow Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PHASE 1: INPUT                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  PDF/Image Transcript                                              │
│        │                                                            │
│        ▼                                                            │
│  ┌─────────────────┐                                               │
│  │ transcript_     │  Extract text (pdfplumber/OCR)                │
│  │ ingestion.py    │  Parse courses (CourseCode, Name, Grade)      │
│  │                 │  Extract student details                      │
│  └────────┬────────┘                                               │
│           │                                                         │
│           ▼                                                         │
│  ┌─────────────────┐                                               │
│  │ course_skill_   │  Load course → skill mappings                 │
│  │ mapping.py      │                                                │
│  └────────┬────────┘                                               │
│           │                                                         │
└───────────┼─────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   PHASE 2: SKILL INFERENCE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────┐                                      │
│  │ skill_aggregation_       │  Map courses → skills                │
│  │ from_parsed.py           │  Calculate skill scores (from grades)│
│  │                          │  Assign skill levels                 │
│  └────────────┬─────────────┘                                      │
│               │                                                     │
│               ▼                                                     │
│  ┌──────────────────────────┐                                      │
│  │ skill_aggregation_       │  Create explainable profile          │
│  │ explainable.py           │  Track evidence (which courses)      │
│  │                          │  Generate explanations               │
│  └────────────┬─────────────┘                                      │
│               │                                                     │
│               ▼                                                     │
│  Output: skill_profile_{student_id}.csv                            │
│                                                                     │
└───────────┬─────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   PHASE 3: QUIZ GENERATION                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Student selects up to 5 skills                                    │
│        │                                                            │
│        ▼                                                            │
│  ┌─────────────────┐                                               │
│  │ quiz_planner.py │  Plan which skills to quiz                    │
│  │                 │  Allocate questions per skill                  │
│  └────────┬────────┘                                               │
│           │                                                         │
│           ▼                                                         │
│  ┌─────────────────┐                                               │
│  │ rag_retriever.py│  Load skill context from corpus               │
│  │                 │  Retrieve relevant chunks                     │
│  └────────┬────────┘                                               │
│           │                                                         │
│           ▼                                                         │
│  ┌──────────────────────┐                                          │
│  │ quiz_generation_     │  Generate MCQ questions                  │
│  │ rag.py / gemini.py   │  Using RAG context + Gemini LLM          │
│  └────────┬─────────────┘                                          │
│           │                                                         │
│           ▼                                                         │
│  ┌─────────────────┐                                               │
│  │ question_       │  Check diversity                              │
│  │ diversity.py    │  Detect duplicates                            │
│  └────────┬────────┘                                               │
│           │                                                         │
│           ▼                                                         │
│  ┌─────────────────┐                                               │
│  │ quiz_validation.│  Validate question quality                    │
│  │ py              │                                                │
│  └────────┬────────┘                                               │
│           │                                                         │
│           ▼                                                         │
│  Output: Quiz JSON (questions, options, answer key)                │
│                                                                     │
└───────────┬─────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      PHASE 4: QUIZ SCORING                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Student submits quiz answers                                      │
│        │                                                            │
│        ▼                                                            │
│  ┌─────────────────┐                                               │
│  │ quiz_scoring.py │  Match answers to answer key                  │
│  │                 │  Calculate scores (overall + per-skill)       │
│  │                 │  Generate performance breakdown               │
│  └────────┬────────┘                                               │
│           │                                                         │
│           ▼                                                         │
│  Output: Quiz results with scores                                  │
│                                                                     │
└───────────┬─────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  PHASE 5: SKILL PROFILE FUSION                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────┐                                      │
│  │ skill_profile_fusion.py  │  Combine transcript scores +         │
│  │                          │  quiz scores                          │
│  │                          │  Weighted fusion algorithm            │
│  └────────────┬─────────────┘                                      │
│               │                                                     │
│               ▼                                                     │
│  Output: Updated skill_profile_with_quiz.csv                       │
│                                                                     │
└───────────┬─────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PHASE 6: JOB MATCHING                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────┐                                      │
│  │ job_postings_ingestion.py│  Load job postings                   │
│  │                          │  Extract skills from descriptions     │
│  └────────────┬─────────────┘                                      │
│               │                                                     │
│               ▼                                                     │
│  ┌──────────────────────────┐                                      │
│  │ job_role_model_dynamic.py│  Create role skill templates         │
│  │                          │  Match student skills to roles        │
│  │                          │  Calculate readiness scores           │
│  │                          │  Identify missing/weak skills         │
│  └────────────┬─────────────┘                                      │
│               │                                                     │
│               ▼                                                     │
│  Output: role_readiness_dynamic.csv                                │
│          role_readiness_details_dynamic.csv                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Complete User Journey Flow

```
User Action                    Backend Process                    Data Flow
───────────────────────────────────────────────────────────────────────────

1. Upload PDF                  transcript_ingestion.py            PDF → text
   └─→ Extract text            (pdfplumber/OCR)                   text → courses
   └─→ Parse courses           (regex patterns)                   courses → CSV

2. View Transcript             Load parsed data                   CSV → API
   └─→ Display courses         API endpoint                       API → Frontend

3. Skills Inferred             skill_aggregation_from_parsed.py   courses → skills
   └─→ Course → Skill Map      course_skill_mapping.py            skills → scores
   └─→ Calculate Scores        (grade-based)                      scores → profile

4. Select Skills (≤5)          User selection                     Frontend state
   └─→ Prepare Quiz            quiz_planner.py                    skills → plan

5. Generate Quiz               rag_retriever.py                   skill → context
   └─→ Retrieve Context        (skill corpus)                     context → LLM
   └─→ Generate Questions      quiz_generation_gemini.py          LLM → questions
   └─→ Validate Questions      question_diversity.py              questions → validated

6. Take Quiz                   Display questions                  API → Frontend
   └─→ Submit Answers          Submit to API                      Frontend → API

7. Score Quiz                  quiz_scoring.py                    answers → scores
   └─→ Calculate Scores        (answer key matching)              scores → breakdown

8. Update Profile              skill_profile_fusion.py            transcript + quiz
   └─→ Fuse Scores             (weighted combination)             → updated profile

9. Match Jobs                  job_role_model_dynamic.py          profile → roles
   └─→ Calculate Readiness     (skill matching)                   roles → recommendations
   └─→ Identify Gaps           (missing skills)                   gaps → suggestions

10. View Recommendations       API endpoint                       API → Frontend
    └─→ Display Roles          Frontend component                 Frontend → UI
    └─→ Show Missing Skills    XAI panel                          explanations
```

---

## 📁 Key Data Structures

### Input Files
```
backend/input/
├── course_skill_mapping.csv      # Course → Skill mappings
├── Job_data.json                 # Raw job postings
├── job_postings_sample.csv       # Processed job postings
└── transcript_data.csv           # Historical transcript data
```

### Processing Files
```
backend/content/
└── skill_corpus.csv              # Skill knowledge base for RAG
```

### Output Files
```
backend/output/
├── transcript_parsed_single.csv          # Parsed transcripts
├── skill_profile_{student_id}.csv        # Initial skill profiles
├── skill_profiles_with_quiz.csv          # Updated after quiz
├── job_role_skill_templates_dynamic.csv  # Role templates
├── role_readiness_dynamic.csv            # Role matching results
├── role_readiness_details_dynamic.csv    # Detailed role analysis
├── quiz_plans.csv                        # Quiz plans
├── quiz_questions_generated.csv          # Generated questions
├── quiz_results_scored.csv               # Quiz scores
└── quizzes/                              # Quiz JSON files
    ├── {quiz_id}_meta.json
    └── {quiz_id}_questions.json
```

---

## 🔌 API Endpoints Overview

```
POST   /upload-transcript              Upload and parse transcript
GET    /students/{id}/skills           Get student skill profile
GET    /students/{id}/roles            Get job recommendations
POST   /prepare-quiz                   Generate quiz for selected skills
POST   /submit-quiz                    Submit quiz answers and get scores
GET    /students/{id}/xai/skills       Get skill explanations
GET    /students/{id}/xai/roles        Get role match explanations
POST   /admin/process-jobs             Process job postings (admin)
```

---

## 🧩 Component Dependencies

```
┌─────────────────────┐
│  transcript_        │
│  ingestion.py       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  course_skill_      │
│  mapping.py         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  skill_aggregation_ │
│  from_parsed.py     │
└──────────┬──────────┘
           │
           ├──────────────────────┐
           │                      │
           ▼                      ▼
┌─────────────────────┐  ┌─────────────────────┐
│  quiz_generation_   │  │  job_role_model_    │
│  rag.py             │  │  dynamic.py         │
└──────────┬──────────┘  └─────────────────────┘
           │
           ▼
┌─────────────────────┐
│  quiz_scoring.py    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  skill_profile_     │
│  fusion.py          │
└─────────────────────┘
```

---

## 🎯 Key Algorithms

### 1. Skill Score Calculation
```
For each course in transcript:
  - Get mapped skills (from course_skill_mapping.csv)
  - Convert grade to numeric score (A=4.0, B=3.0, etc.)
  - Weight by skill level (MainSkill gets higher weight)
  - Aggregate across all courses
  - Normalize to 0-1 range
```

### 2. Quiz Generation (RAG)
```
1. Retrieve relevant chunks from skill_corpus.csv
2. Combine chunks into context
3. Prompt Gemini LLM with context + skill name
4. Generate MCQ questions (question + 4 options)
5. Validate diversity (no duplicate questions)
6. Validate quality (format, clarity)
```

### 3. Skill Profile Fusion
```
For each skill:
  transcript_score = score from course grades
  quiz_score = score from quiz performance
  final_score = w1 * transcript_score + w2 * quiz_score
  where w1 + w2 = 1.0 (typically w1=0.6, w2=0.4)
```

### 4. Role Readiness Calculation
```
For each role:
  1. Get required skills (from role template)
  2. Match student skills to required skills
  3. Calculate coverage = (matched_skills / total_required) * 100
  4. Calculate readiness = weighted average of matched skill scores
  5. Identify missing/weak skills (score < threshold)
```

---

## 🔐 Security & Validation

```
Quiz Security:
├── Time limits (session tokens)
├── Answer key encryption
├── Response validation
└── Violation detection (tab switching, etc.)

Data Validation:
├── File type validation (PDF/images)
├── Student ID extraction validation
├── Grade format validation
└── Question format validation
```

---

## 📊 Explainability (XAI)

### Skill Evidence
- Which courses contributed to each skill
- Grade-based evidence
- Skill level justification

### Role Evidence
- Required skills for role
- Student's skill scores
- Missing/weak skills identification
- Readiness score breakdown

---

## 🚀 Extension Points

### Easy to Extend:
1. **New quiz generation methods** - Add new files similar to `quiz_generation_gemini.py`
2. **New skill aggregation methods** - Add new aggregation algorithms
3. **New job matching algorithms** - Modify `job_role_model_dynamic.py`
4. **New frontend components** - Add to `frontend-react/src/components/`

### Requires Architecture Changes:
1. **Database integration** - Currently file-based (CSV/JSON)
2. **User authentication** - Currently stateless
3. **Real-time updates** - Currently request-response
4. **Batch processing** - Currently single-student focused

---

This architecture overview provides a high-level view. For detailed implementation, refer to `PROJECT_UNDERSTANDING_GUIDE.md` for the complete reading order.

