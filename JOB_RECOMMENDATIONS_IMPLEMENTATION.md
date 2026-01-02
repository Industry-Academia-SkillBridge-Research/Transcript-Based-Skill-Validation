# Job Recommendations & React Router Implementation

## Overview

This document describes the implementation of:
1. **React Router Integration** - Proper URL routing for navigation
2. **Job Recommendations System** - Display matched jobs based on student skills
3. **Job Processing Endpoint** - Process Job_data.json to create job roles

---

## ✅ 1. React Router Integration

### What Was Done

- ✅ Installed `react-router-dom` package
- ✅ Wrapped `App` component with `BrowserRouter` in `main.jsx`
- ✅ Added `useNavigate` and `useLocation` hooks to `App.jsx`

### URL Routing

The app now supports proper URL routing. However, to keep backward compatibility and simplicity, we're still using state-based navigation internally. The Router is set up for future URL-based routes if needed.

**Note**: Currently, URLs still show `localhost:5173` for all pages. To enable actual route URLs (e.g., `/upload`, `/quiz`, `/dashboard`), you would need to refactor the component rendering logic to use `<Routes>` and `<Route>` components. This is optional and can be done later if needed.

---

## ✅ 2. Job Recommendations Component

### Component: `JobRecommendations.jsx`

**Location**: `frontend-react/src/components/JobRecommendations.jsx`

**Features**:
- Displays job roles matched to student's verified skill profile
- Shows readiness score (0-100%) for each role
- Displays skill coverage percentage
- Lists missing/weak skills for each role
- Expandable skill breakdown with detailed analysis
- Color-coded readiness indicators:
  - **Green** (≥80%): Excellent Match
  - **Blue** (≥60%): Good Match
  - **Yellow** (≥40%): Moderate Match
  - **Red** (<40%): Low Match

**Data Displayed**:
- Role name
- Readiness score (weighted skill match)
- Skill coverage (percentage of required skills present)
- Missing/weak skills list
- Detailed skill breakdown (when expanded):
  - Required importance weight
  - Student's skill score
  - Student's skill level
  - Weak/missing indicator

---

## ✅ 3. Backend Job Processing Endpoint

### Endpoint: `POST /admin/process-jobs`

**Location**: `backend/src/api/main.py`

**What It Does**:
1. Converts `Job_data.json` → `job_postings_sample.csv` (if JSON exists)
2. Processes job postings and extracts skills from descriptions
3. Creates role skill templates (`job_role_skill_templates_dynamic.csv`)
4. Computes role readiness for all students (`role_readiness_dynamic.csv`)

**Files Created**:
- `output/job_postings_sample.csv`
- `output/job_skill_matches.csv`
- `output/job_role_skill_templates_dynamic.csv`
- `output/role_readiness_dynamic.csv`
- `output/role_readiness_explainable.csv`

**Usage**:
```javascript
import { processJobs } from "./api";

// Process jobs (usually called once when setting up the system)
const result = await processJobs();
console.log(result.message); // "Jobs processed successfully..."
```

---

## ✅ 4. Navigation Flow

### User Journey:

1. **Upload Transcript** → Shows transcript details
2. **View Skills** → Shows inferred skills from transcript
3. **Select Skills & Generate Quiz** → Quiz page
4. **Complete Quiz** → Quiz results page
5. **View Skill Profile Dashboard** → Shows transcript + quiz scores
6. **View Job Recommendations** → Shows matched job roles ✨ **NEW**

### Navigation Path:

```
Quiz Results → Skill Dashboard → Job Recommendations
```

---

## ✅ 5. Job Matching Algorithm

### How It Works:

1. **Job Ingestion**:
   - Load jobs from `Job_data.json` or `job_postings_sample.csv`
   - Extract skills from job descriptions using phrase-based matching
   - Create `(JobID, RoleName, Skill)` mappings

2. **Role Canonicalization**:
   - Normalize job titles (e.g., "Data Scientist", "ML Engineer")
   - Map similar roles to canonical names

3. **Role Skill Templates**:
   - For each `(RoleName, Skill)` pair:
     - Calculate `Support` = frequency of skill in role
     - Calculate `Importance` = `Support × log(1 + RolePostingCount)`
     - Normalize importance to [0, 1] per role

4. **Readiness Calculation**:
   - For each `(StudentID, RoleName)`:
     - Get student's verified skills (transcript + quiz fused)
     - Compute weighted overlap: `Σ(Importance × StudentScore) / Σ(Importance)`
     - Identify weak/missing skills (score < 0.4 threshold)
     - Calculate skill coverage (skills present / skills required)

---

## ✅ 6. Files Modified/Created

### Frontend:
1. ✅ `frontend-react/src/components/JobRecommendations.jsx` - New component
2. ✅ `frontend-react/src/components/SkillProfileDashboard.jsx` - Added "View Job Recommendations" button
3. ✅ `frontend-react/src/App.jsx` - Integrated JobRecommendations component
4. ✅ `frontend-react/src/main.jsx` - Added BrowserRouter wrapper
5. ✅ `frontend-react/src/api.js` - Added `processJobs()` function

### Backend:
1. ✅ `backend/src/api/main.py` - Added `POST /admin/process-jobs` endpoint

---

## 📋 Setup Instructions

### 1. Process Jobs (One-Time Setup)

Before students can see job recommendations, you need to process the job data:

**Option A: Via API** (Recommended)
```bash
curl -X POST http://localhost:8000/admin/process-jobs
```

**Option B: Via Python Script**
```bash
cd backend
python src/convert_job_json_to_csv.py  # Convert JSON to CSV
python src/job_postings_ingestion.py   # Extract skills and create templates
python src/job_role_model_dynamic.py   # Compute role readiness
```

### 2. Ensure Job Data Exists

Make sure `backend/input/Job_data.json` exists with job postings data. The JSON should have this structure:
```json
[
  {
    "job_id": "1",
    "title": "Data Scientist",
    "company": "Tech Corp",
    "location": "Colombo",
    "description": "We are looking for a data scientist with Python, Machine Learning, SQL skills...",
    "job_url": "https://linkedin.com/jobs/...",
    "posted_date": "2024-01-01"
  },
  ...
]
```

### 3. Student Flow

1. Student uploads transcript
2. Student completes quiz (optional, but improves skill scores)
3. Student views skill dashboard
4. Student clicks "View Job Recommendations"
5. System shows matched job roles based on verified skills

---

## 🎯 Features

### Job Recommendations Page

- **Summary Statistics**:
  - Total roles matched
  - Top match percentage
  - Average readiness across all roles

- **Role Cards**:
  - Role name
  - Readiness score with color coding
  - Skill coverage percentage
  - Missing/weak skills highlighted

- **Detailed View**:
  - Expandable skill breakdown
  - Required importance vs student score
  - Skill level indicators
  - Weak/missing skill flags

---

## 🔧 API Endpoints

### Get Job Recommendations
```http
GET /students/{student_id}/roles
```

**Response**:
```json
{
  "student_id": "IT21013928",
  "source_file": "role_readiness_dynamic.csv",
  "roles": [
    {
      "RoleName": "Data Scientist",
      "ReadinessScore": 0.85,
      "Coverage": 0.90,
      "NumSkills": 10,
      "NumSkillsPresent": 9,
      "NumWeakOrMissing": 1,
      "WeakOrMissingSkills": "Big Data Processing"
    },
    ...
  ]
}
```

### Get Role Details (Skill Breakdown)
```http
GET /students/{student_id}/xai/roles?role=Data Scientist
```

**Response**:
```json
{
  "student_id": "IT21013928",
  "role_name": "Data Scientist",
  "num_required_skills": 10,
  "num_weak_or_missing": 1,
  "required_skills": [
    {
      "skill": "Machine Learning",
      "required_importance": 1.0,
      "student_score": 0.92,
      "student_level": "Advanced",
      "attained_fraction": 0.92,
      "is_weak_or_missing": false
    },
    ...
  ]
}
```

### Process Jobs
```http
POST /admin/process-jobs
```

**Response**:
```json
{
  "status": "success",
  "message": "Jobs processed successfully...",
  "files_created": [...]
}
```

---

## 🚀 Next Steps (Optional Enhancements)

1. **Filter & Sort**:
   - Filter roles by readiness threshold
   - Sort by readiness, coverage, or role name

2. **Export**:
   - Export job recommendations as PDF/CSV

3. **Job Details**:
   - Link to actual job postings
   - Show job description, company, location

4. **Recommendations**:
   - AI-powered personalized job suggestions
   - Career path recommendations based on skill gaps

5. **URL Routing**:
   - Implement actual routes (e.g., `/dashboard`, `/jobs`)
   - Enable bookmarkable pages

---

## 📝 Notes

- Job roles are created automatically from `Job_data.json`
- Role matching uses verified skills (transcript + quiz fused scores)
- Missing skills are skills with score < 0.4 (configurable threshold)
- Role readiness is computed as weighted skill overlap
- The system supports any number of jobs and roles

---

## ✅ Testing Checklist

- [x] Job processing endpoint works
- [x] Job recommendations display correctly
- [x] Skill breakdown expands/collapses
- [x] Navigation from dashboard to jobs works
- [x] Readiness scores are calculated correctly
- [x] Missing skills are highlighted
- [x] React Router is installed and configured

