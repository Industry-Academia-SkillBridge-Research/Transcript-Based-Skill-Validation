# Project Understanding Guide - Reading Order

This guide provides a **systematic file reading order** to deeply understand the Transcript-Based Skill Validation and Job Alignment project, from high-level concepts to detailed implementation.

---

## 📚 **PHASE 1: Project Overview & Architecture** (Start Here)

### 1.1 High-Level Understanding
**Read these files first to understand what the project does:**

1. **`PROJECT_STATUS_ASSESSMENT.md`** ⭐ START HERE
   - Complete project overview
   - Research problem and workflow
   - Component status summary
   - What's implemented vs. what's missing

2. **`backend/README.md`**
   - Backend architecture overview
   - Project structure
   - Main pipeline workflow

3. **`RUN_INSTRUCTIONS.md`**
   - How to set up and run the project
   - Prerequisites
   - Understanding the execution flow

4. **`COMPLETE_SETUP_GUIDE.md`** (if exists)
   - Detailed setup instructions
   - Environment configuration

---

## 📊 **PHASE 2: Data Flow & Core Concepts**

### 2.1 Understanding Data Flow
**Understand how data moves through the system:**

5. **`TRANSCRIPT_INGESTION_STATUS.md`**
   - How transcripts are processed
   - Course extraction logic
   - Data structures used

6. **`SKILL_INFERENCE_STATUS.md`**
   - How skills are inferred from courses
   - Skill scoring methodology
   - Skill level assignment

7. **`SKILL_FUSION_STATUS.md`**
   - How transcript scores and quiz scores are combined
   - Fusion algorithm details

8. **`ROLE_MATCHING_STATUS.md`**
   - Job matching algorithm
   - Role readiness calculation
   - Skill gap analysis

---

## 🔧 **PHASE 3: Backend Core Implementation**

### 3.1 Data Ingestion Layer
**Start with data input processing:**

9. **`backend/src/transcript_loader.py`**
   - CSV loading utilities
   - Data normalization

10. **`backend/src/transcript_ingestion.py`**
    - PDF/image parsing
    - Text extraction (PDF plumber + OCR)
    - Course extraction logic
    - Student details extraction

11. **`backend/src/course_skill_mapping.py`**
    - Course-to-skill mapping loader
    - Mapping data structure
    - How courses map to skills

12. **`backend/input/course_skill_mapping.csv`** (Review structure)
    - Actual mapping data
    - Understanding the mapping format

### 3.2 Skill Processing Layer
**How skills are derived and scored:**

13. **`backend/src/skill_aggregation_from_parsed.py`**
    - Skill inference from parsed courses
    - Skill score calculation
    - Basic skill profile creation

14. **`backend/src/skill_aggregation_explainable.py`**
    - Explainable skill aggregation
    - Evidence tracking
    - Skill level assignment (Beginner/Intermediate/Advanced)

15. **`backend/src/skill_profile_fusion.py`**
    - Combining transcript-based and quiz-based scores
    - Fusion algorithm implementation
    - Final skill profile generation

### 3.3 Job Processing Layer
**How jobs are processed and matched:**

16. **`backend/src/job_postings_ingestion.py`**
    - Job data ingestion
    - Skill extraction from job descriptions
    - Job-to-skill mapping

17. **`backend/src/job_role_model_dynamic.py`**
    - Role template creation
    - Role readiness calculation
    - Skill gap identification
    - Missing skills detection

18. **`backend/src/convert_job_json_to_csv.py`** (if using JSON input)
    - JSON to CSV conversion
    - Data format transformation

---

## 📝 **PHASE 4: Quiz Generation System**

### 4.1 Quiz Planning
**How quizzes are planned:**

19. **`backend/src/quiz_planner.py`**
    - Quiz planning logic
    - Skill selection for quizzes
    - Question allocation

20. **`QUIZ_SETUP_GUIDE.md`**
    - Quiz setup requirements
    - API key configuration

### 4.2 Quiz Generation
**How quiz questions are generated:**

21. **`backend/docs/RAG_RETRIEVAL.md`**
    - RAG (Retrieval-Augmented Generation) concepts
    - How the skill corpus is used

22. **`backend/src/build_skill_corpus_chunks.py`**
    - Building the skill corpus
    - Chunking strategy

23. **`backend/src/rag_retriever.py`**
    - RAG retrieval implementation
    - Context loading for quiz generation

24. **`backend/src/rag_retrieval.py`**
    - Retrieval logic
    - Similarity search

25. **`backend/src/quiz_generation_rag.py`**
    - RAG-based quiz generation
    - Question generation using retrieved context

26. **`backend/src/quiz_generation_gemini.py`**
    - Gemini LLM integration
    - MCQ generation logic

27. **`backend/src/question_diversity.py`**
    - Question diversity checking
    - Duplicate detection

28. **`backend/src/quiz_validation.py`**
    - Question validation
    - Quality checks

29. **`backend/docs/QUIZ_STORAGE.md`**
    - Quiz storage structure
    - Quiz metadata

### 4.3 Quiz Scoring
**How quizzes are scored:**

30. **`backend/src/quiz_scoring.py`**
    - Answer key matching
    - Score calculation
    - Per-skill performance breakdown

31. **`QUIZ_SCORING_STATUS.md`**
    - Scoring system overview

---

## 🌐 **PHASE 5: API Layer**

### 5.1 Main API Implementation
**The REST API that connects everything:**

32. **`backend/src/api/main.py`** ⭐ CRITICAL FILE
    - All API endpoints
    - Request/response models
    - Endpoint implementations:
      - `/upload-transcript` - Transcript upload
      - `/students/{id}/skills` - Get student skills
      - `/students/{id}/roles` - Get job recommendations
      - `/prepare-quiz` - Generate quiz
      - `/submit-quiz` - Submit quiz answers
      - `/students/{id}/xai/skills` - Skill explanations
      - `/students/{id}/xai/roles` - Role explanations
    - Read this file section by section focusing on each endpoint

---

## 💻 **PHASE 6: Frontend Implementation**

### 6.1 Frontend Architecture
**How the user interface works:**

33. **`frontend-react/src/App.jsx`** ⭐ MAIN FRONTEND FILE
    - Overall application structure
    - State management
    - Navigation flow
    - Component orchestration

34. **`frontend-react/src/api.js`**
    - API client functions
    - All backend API calls
    - Request/response handling

### 6.2 Frontend Components
**Individual UI components (read in order of user flow):**

35. **`frontend-react/src/components/FileUpload.jsx`**
    - File upload UI

36. **`frontend-react/src/components/UploadTranscript.jsx`**
    - Transcript upload component

37. **`frontend-react/src/components/TranscriptDetailsPage.jsx`**
    - Transcript display after upload
    - Course listing
    - Student details

38. **`frontend-react/src/components/SkillsPage.jsx`**
    - Skills display
    - Skill selection UI (up to 5 skills)

39. **`frontend-react/src/components/QuizPage.jsx`**
    - Quiz taking interface
    - Question display
    - Timer implementation

40. **`frontend-react/src/components/QuizResultPage.jsx`**
    - Quiz results display
    - Score breakdown
    - Navigation to next steps

41. **`frontend-react/src/components/SkillProfileDashboard.jsx`**
    - Combined skill profile view
    - Transcript + quiz scores
    - Visual dashboard

42. **`frontend-react/src/components/JobRecommendations.jsx`**
    - Job/role recommendations display
    - Role matching results
    - Missing skills visualization

43. **`frontend-react/src/components/RoleMatches.jsx`**
    - Role matching component

44. **`frontend-react/src/components/XaiPanel.jsx`**
    - Explainability panel
    - Skill and role evidence display

45. **`frontend-react/src/utils/quizSecurity.js`**
    - Quiz security measures
    - Time validation
    - Violation detection

---

## 📖 **PHASE 7: Supporting Documentation**

### 7.1 Feature-Specific Documentation
**Deep dive into specific features:**

46. **`backend/docs/BUILD_SKILL_CORPUS.md`**
    - How to build the skill corpus
    - Corpus structure

47. **`backend/CORPUS_QUICKSTART.md`**
    - Quick guide to skill corpus

48. **`backend/docs/QUESTION_DIVERSITY.md`**
    - Question diversity strategy

49. **`backend/docs/QUIZ_VALIDATION.md`**
    - Quiz validation methodology

50. **`backend/docs/SECURITY_FAIRNESS.md`**
    - Security and fairness considerations

51. **`SECURITY_FAIRNESS_SUMMARY.md`**
    - Security summary

52. **`QUESTION_DIVERSITY_SUMMARY.md`**
    - Diversity summary

### 7.2 Implementation Status Documents
**Current state of various components:**

53. **`SKILL_SELECTION_STATUS.md`**
    - Skill selection implementation

54. **`RAG_QUIZ_GENERATION_STATUS.md`**
    - RAG quiz generation status

55. **`QUIZ_SECURITY_CONTROLS_STATUS.md`**
    - Security controls status

56. **`JOB_RECOMMENDATIONS_IMPLEMENTATION.md`**
    - Job recommendations implementation

57. **`SKILL_DASHBOARD_IMPLEMENTATION.md`**
    - Dashboard implementation

58. **`ROUTING_EXPLANATION.md`**
    - Routing flow explanation

---

## 🔄 **PHASE 8: Pipeline & Orchestration**

### 8.1 Full Pipeline
**How everything fits together:**

59. **`backend/src/run_full_pipeline_for_student.py`** ⭐ PIPELINE ORCHESTRATION
    - Complete pipeline execution
    - Pre-quiz and post-quiz phases
    - Step-by-step workflow

60. **`backend/VALIDATION_SUMMARY.md`**
    - Validation summary

61. **`QUIZ_TROUBLESHOOTING.md`**
    - Common issues and solutions

---

## 📁 **PHASE 9: Data Structures & Configuration**

### 9.1 Input Data
**Understanding input data formats:**

62. **`backend/input/course_skill_mapping.csv`** (Review samples)
    - Course-skill mappings

63. **`backend/input/Job_data.json`** (Review structure)
    - Job posting data structure

64. **`backend/configs/job_role_templates.json`** (Review structure)
    - Role template configuration

65. **`backend/content/skill_corpus.csv`** (Review samples)
    - Skill knowledge corpus

### 9.2 Output Data
**Understanding output data structures:**

66. Review sample output files in `backend/output/`:
    - `skill_profile_*.csv` - Skill profiles
    - `role_readiness_dynamic.csv` - Role matching results
    - `quiz_*.json` - Quiz data
    - `quiz_results_scored.csv` - Quiz scores

---

## 🛠️ **PHASE 10: Utility & Supporting Code**

### 10.1 Utility Scripts
**Supporting utilities:**

67. **`backend/src/make_skill_keys.py`**
    - Skill key generation

68. **`backend/src/generate_skill_corpus_template.py`**
    - Corpus template generation

69. **`backend/src/auto_fill_skill_corpus.py`**
    - Auto-filling skill corpus

70. **`backend/src/add_corpus_chunks.py`**
    - Adding corpus chunks

71. **`backend/src/merge_parsed_student_into_profiles.py`**
    - Merging student data

---

## 📋 **Quick Reference: Critical Files Summary**

### Must-Read Files (In Order):
1. `PROJECT_STATUS_ASSESSMENT.md` - Overall understanding
2. `backend/src/api/main.py` - API endpoints (read section by section)
3. `backend/src/transcript_ingestion.py` - Transcript processing
4. `backend/src/skill_aggregation_from_parsed.py` - Skill inference
5. `backend/src/quiz_generation_rag.py` - Quiz generation
6. `backend/src/quiz_scoring.py` - Quiz scoring
7. `backend/src/job_role_model_dynamic.py` - Job matching
8. `frontend-react/src/App.jsx` - Frontend flow
9. `backend/src/run_full_pipeline_for_student.py` - Full pipeline

---

## 🎯 **Recommended Reading Strategy**

### For Quick Understanding (2-3 hours):
1. Read Phase 1 (Overview & Architecture)
2. Read Phase 2 (Data Flow)
3. Read `backend/src/api/main.py` (API endpoints)
4. Read `frontend-react/src/App.jsx` (Frontend flow)

### For Deep Understanding (1-2 days):
1. Follow all phases in order
2. Read code files along with their corresponding status documents
3. Review sample input/output data files
4. Trace through a complete user journey in the code

### For Implementation/Modification:
1. Complete all phases
2. Focus deeply on the module you want to modify
3. Understand dependencies (what calls what)
4. Review test files if they exist

---

## 🔍 **Understanding Dependencies**

### Key Dependency Flow:
```
transcript_ingestion.py
    ↓
skill_aggregation_from_parsed.py
    ↓
skill_profile_fusion.py (after quiz)
    ↓
job_role_model_dynamic.py
    ↓
API endpoints (main.py)
    ↓
Frontend (App.jsx → Components)
```

### Quiz Generation Flow:
```
quiz_planner.py
    ↓
rag_retriever.py / rag_retrieval.py
    ↓
quiz_generation_rag.py / quiz_generation_gemini.py
    ↓
question_diversity.py
    ↓
quiz_validation.py
    ↓
quiz_scoring.py
```

---

## 💡 **Tips for Understanding**

1. **Start with the big picture** - Don't dive into code immediately
2. **Follow the user journey** - Trace what happens when a user uploads a transcript
3. **Use the status documents** - They provide context for each component
4. **Read API endpoints carefully** - They show how components connect
5. **Review data structures** - Understanding CSV/JSON structures helps understand the code
6. **Test as you read** - Run the application and observe behavior while reading code

---

## 🚀 **Next Steps After Understanding**

Once you understand the project:
1. Identify areas you want to modify or extend
2. Review the "MISSING INTEGRATIONS" section in PROJECT_STATUS_ASSESSMENT.md
3. Check TODO comments in the code
4. Review the validation and status documents for known issues
5. Consider adding tests or documentation

---

**Good luck with your deep dive! 🎓**

