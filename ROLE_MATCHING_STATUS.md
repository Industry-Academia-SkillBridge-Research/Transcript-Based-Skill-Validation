# Role Matching Using Real Job Data - Implementation Status

## ✅ REQUIREMENT 9: Role Matching Using Real Job Data

**Goal**: Match student verified skill profiles with real job role requirements to recommend suitable roles and identify skill gaps.

**Research Importance**: Connects validated student skills to real-world job opportunities, providing actionable career guidance.

---

## Status: ✅ **FULLY IMPLEMENTED** (with minor enhancements possible)

---

## ✅ 9.1 Job Ingestion (JSON → CSV → Structured Postings)

### Status: ✅ **COMPLETE**

**Implementation:**

1. **JSON → CSV Conversion**
   - ✅ Script: `backend/src/convert_job_json_to_csv.py`
   - ✅ Input: `input/Job_data.json`
   - ✅ Output: `input/job_postings_sample.csv`
   - ✅ Extracts: JobID, Title, Company, Location, Description, Source, PostedDate

2. **Job Posting Processing**
   - ✅ Script: `backend/src/job_postings_ingestion.py`
   - ✅ Loads CSV job postings
   - ✅ Builds skill vocabulary from course-skill mapping
   - ✅ Extracts skills from job descriptions using phrase-based matching
   - ✅ Creates `(JobID, RoleName, Skill)` table

3. **Role Canonicalization**
   - ✅ Normalizes noisy job titles (e.g., "data scientist", "ml engineer")
   - ✅ Maps to canonical role names (e.g., "Data Scientist", "ML Engineer")

**Files:**
- `backend/src/convert_job_json_to_csv.py` - JSON → CSV converter
- `backend/src/job_postings_ingestion.py` - Job processing and skill extraction
- `input/Job_data.json` - Raw job data (JSON format)
- `input/job_postings_sample.csv` - Processed job postings (CSV format)

**Manual Setup:**
- ⚠️ **YOU NEED TO RUN** `convert_job_json_to_csv.py` if you update `Job_data.json`:
  ```bash
  cd backend
  python src/convert_job_json_to_csv.py
  ```

---

## ✅ 9.2 Skill Extraction from Job Descriptions

### Status: ✅ **COMPLETE**

**Implementation:**

- ✅ **Skill Vocabulary**: Built from `course_skill_mapping.csv`
- ✅ **Matching Method**: Phrase-based text matching (checks if all skill tokens appear in job description)
- ✅ **Output**: `output/job_skill_matches.csv` (JobID, RoleName, Skill)

**Current Matching:**
- Simple phrase-based matching (token-level)
- Future enhancement: Can upgrade to embeddings/NLP for better semantic matching

**Files:**
- `backend/src/job_postings_ingestion.py` - `extract_skills_from_job()` function

---

## ✅ 9.3 Role Skill Templates Creation

### Status: ✅ **COMPLETE**

**Implementation:**

For each `(RoleName, Skill)` pair, computes:

1. **JobCount**: Number of jobs for that role where the skill appeared
2. **RolePostingCount**: Total number of jobs for that role
3. **Support**: `JobCount / RolePostingCount` (frequency of skill in role)
4. **Importance**: `Support × log(1 + RolePostingCount)` (weighted by role popularity)
5. **ImportanceNorm**: Normalized importance [0, 1] per role (for easier interpretation)

**Formula:**
```
Importance = (JobCount / RolePostingCount) × log(1 + RolePostingCount)
ImportanceNorm = Importance / max(Importance) per role
```

**Output:**
- ✅ `output/job_role_skill_templates_dynamic.csv`
  - Columns: RoleName, Skill, JobCount, RolePostingCount, Support, Importance, ImportanceNorm

**Files:**
- `backend/src/job_postings_ingestion.py` - `build_role_skill_templates()` function

---

## ✅ 9.4 Role Matching (Readiness Score Calculation)

### Status: ✅ **COMPLETE**

**Implementation:**

For each `(StudentID, RoleName)` pair:

1. **Get Role Requirements**: All skills for the role with importance weights
2. **Get Student Skills**: Verified skill profile (transcript + quiz fused)
3. **Compute Weighted Overlap**:
   ```
   attained_weighted = Σ(ImportanceNorm × StudentScore) for all role skills
   total_importance = Σ(ImportanceNorm) for all role skills
   ReadinessScore = attained_weighted / total_importance
   ```
4. **Compute Coverage**: `NumSkillsPresent / NumSkills`
5. **Identify Weak/Missing Skills**: Skills where `StudentScore < 0.4` threshold

**Formula:**
```
ReadinessScore = Σ(ImportanceNorm × StudentScore) / Σ(ImportanceNorm)
```

**Output Files:**
- ✅ `output/role_readiness_dynamic.csv` (Summary)
  - Columns: StudentID, RoleName, ReadinessScore, Coverage, NumSkills, NumSkillsPresent, NumWeakOrMissing, WeakOrMissingSkills (comma-separated list)
- ✅ `output/role_readiness_explainable.csv` (Detailed)
  - Columns: StudentID, RoleName, Skill, RequiredImportance, StudentScore, StudentLevel, AttainedFraction, IsWeakOrMissing

**Files:**
- `backend/src/job_role_model_dynamic.py` - `compute_role_readiness()` function

**Priority for Skill Profiles:**
- Prefers `skill_profiles_with_quiz.csv` (fused profile) over `skill_profiles_explainable.csv` (transcript only)

---

## ✅ 9.5 API Endpoints

### Status: ✅ **COMPLETE**

**Endpoints:**

1. **Get Student Roles** (Summary)
   ```
   GET /students/{student_id}/roles
   ```
   - Returns: List of roles sorted by ReadinessScore (descending)
   - Fields: role_name, readiness_score, coverage, num_skills, num_skills_present, num_weak_or_missing, weak_or_missing_skills

2. **Get Role Details** (Skill-Level Gaps)
   ```
   GET /students/{student_id}/xai/roles?role={role_name}&top_n={top_n}
   ```
   - Returns: Detailed skill breakdown for a role
   - Fields: role_name, num_required_skills, num_weak_or_missing, required_skills (list with skill, required_importance, student_score, student_level, is_weak_or_missing)

**Files:**
- `backend/src/api/main.py` - `get_student_roles()` and `xai_roles()` endpoints

---

## ✅ 9.6 Frontend Display

### Status: ✅ **COMPLETE** (with minor enhancement possible)

**Implementation:**

1. **Recommended Roles Panel**
   - ✅ Displays top roles sorted by readiness score
   - ✅ Shows: Role name, Readiness score, Coverage, Skills present, Weak/missing count
   - ✅ "Explain" button opens XAI panel with detailed skill gaps

2. **XAI Panel (Detailed View)**
   - ✅ Shows all required skills for a role
   - ✅ Highlights weak/missing skills (red badge: "gap")
   - ✅ Shows student score vs. required importance
   - ✅ Displays student level for each skill

**Current Display:**
```
Role Suggestions Panel:
- Role Name
- Readiness Score (%)
- Coverage (%)
- Skills Present (X/Y)
- Weak/Missing Count
- [Explain Button] → Opens XAI panel
```

**XAI Panel (when clicking "Explain"):**
```
Role Evidence:
- Weak or missing: X / Y skills
- List of all skills:
  - Skill name
  - Required importance
  - Student score
  - Student level
  - [gap/ok badge]
```

**Files:**
- `frontend-react/src/App.jsx` - Role suggestions panel
- `frontend-react/src/components/RoleMatches.jsx` - Role matches table
- `frontend-react/src/components/XaiPanel.jsx` - Detailed skill gap view

**Minor Enhancement (Optional):**
- ⚠️ **COULD ENHANCE**: Show missing skills list inline in the role card (not just in XAI panel)
  - Currently: Missing skills count is shown, full list only in XAI panel
  - Enhancement: Expandable section in role card showing top missing skills

---

## ✅ Complete Pipeline Flow

### Step-by-Step:

1. **Convert Job Data** (if needed)
   ```bash
   python backend/src/convert_job_json_to_csv.py
   ```
   - Converts `Job_data.json` → `job_postings_sample.csv`

2. **Process Job Postings**
   ```bash
   python backend/src/job_postings_ingestion.py
   ```
   - Extracts skills from job descriptions
   - Creates role skill templates
   - Outputs: `job_role_skill_templates_dynamic.csv`

3. **Run Role Matching**
   ```bash
   python backend/src/job_role_model_dynamic.py
   ```
   - Computes readiness scores for all students
   - Outputs: `role_readiness_dynamic.csv` and `role_readiness_explainable.csv`

4. **View in Frontend**
   - Student uploads transcript → skills inferred
   - Student takes quiz → skills fused
   - Frontend calls `/students/{student_id}/roles` → shows recommended roles
   - Student clicks "Explain" → shows detailed skill gaps

---

## 📋 Summary

### ✅ What's Implemented:

1. ✅ JSON → CSV conversion (`convert_job_json_to_csv.py`)
2. ✅ Job posting processing and skill extraction
3. ✅ Role skill templates with importance weights
4. ✅ Role matching with weighted readiness score
5. ✅ Missing/weak skills identification
6. ✅ API endpoints for role recommendations
7. ✅ Frontend display of recommended roles
8. ✅ Detailed skill gap view (XAI panel)

### ⚠️ Manual Steps Required:

1. **Convert Job Data** (when `Job_data.json` is updated):
   ```bash
   python backend/src/convert_job_json_to_csv.py
   ```

2. **Process Job Postings** (when job CSV is updated):
   ```bash
   python backend/src/job_postings_ingestion.py
   ```

3. **Run Role Matching** (after skill profiles are updated):
   ```bash
   python backend/src/job_role_model_dynamic.py
   ```
   - This should be run after:
     - Transcript upload (for baseline profiles)
     - Quiz submission (for fused profiles)

### 🔄 Auto-Integration Opportunity:

- ⚠️ **POTENTIAL ENHANCEMENT**: Auto-trigger role matching after quiz submission
  - Currently: Manual step
  - Enhancement: Add role matching to `submit_quiz` endpoint after skill fusion

### 📝 Minor Enhancement (Optional):

- **Inline Missing Skills Display**: Show top missing skills directly in role card (expandable)
  - Currently: Count only, full list in XAI panel
  - Benefit: Faster visibility without clicking "Explain"

---

## ✅ Status: **FULLY FUNCTIONAL**

The role matching system is complete and functional. All core requirements are met:
- ✅ Job data ingestion (JSON → CSV)
- ✅ Skill extraction from job descriptions
- ✅ Role skill templates with importance weights
- ✅ Readiness score calculation
- ✅ Missing skills identification
- ✅ API endpoints
- ✅ Frontend display with detailed gap analysis

The system successfully connects validated student skills to real-world job opportunities, providing actionable career guidance.

