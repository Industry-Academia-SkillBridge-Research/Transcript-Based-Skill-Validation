# Skill Inference Pipeline - Implementation Status

## ✅ REQUIREMENT 3: Skill Inference (Courses → Skills with Scores)

**Goal**: Convert course rows into a skill profile with year-weighted scoring.

---

## Status: ✅ **FULLY IMPLEMENTED** (enhanced with year weights)

### ✅ Step 1: Course → Skill Mapping
**Status**: ✅ **COMPLETE**

- ✅ Loads `input/course_skill_mapping.csv`
- ✅ Maps each course to skills (Skill1-5, MainSkill)
- ✅ Gets skill list per course

**Files:**
- `backend/src/course_skill_mapping.py` - Mapping loader
- `backend/input/course_skill_mapping.csv` - Mapping data

---

### ✅ Step 2: Score Contribution Calculation
**Status**: ✅ **NOW ENHANCED WITH YEAR WEIGHTS**

**What's calculated:**
- ✅ **Grade point**: 0-4.0 (A+ = 4.0, F = 0.0)
- ✅ **Year weight**: 
  - Year 1: 0.8 (foundation)
  - Year 2: 1.0 (core, baseline)
  - Year 3: 1.1 (advanced)
  - Year 4: 1.2 (specialization, highest weight)
- ✅ **Course contribution**: `GradePoint × YearWeight`
- ✅ **Per-skill contribution**: Distributed evenly across all skills for a course

**Formula:**
```
Contribution = GradePoint × YearWeight
Per-Skill Contribution = Contribution / Number of Skills
```

**Example:**
- Course: IT4100 (Year 4), Grade: A (4.0)
- Year Weight: 1.2
- Skills: ["Machine Learning", "Deep Learning"]
- Contribution: 4.0 × 1.2 = 4.8
- Per-skill: 4.8 / 2 = 2.4 each

**Files:**
- `backend/src/skill_aggregation_from_parsed.py` - Enhanced with year weights

---

### ✅ Step 3: Aggregation Per Student
**Status**: ✅ **COMPLETE**

**Output per (StudentID, Skill):**
- ✅ **EvidenceCount**: Number of courses contributing to this skill
- ✅ **TotalContribution**: Sum of all course contributions
- ✅ **ScoreNormalized**: Normalized to [0, 1] range
  - Formula: `Average Contribution / MAX_COURSE_SCORE (4.8)`
  - Clipped to [0, 1]
- ✅ **SkillLevel**: 
  - Advanced (≥ 0.75)
  - Proficient (≥ 0.50)
  - Developing (≥ 0.25)
  - Beginner (< 0.25)

**Files:**
- `backend/src/skill_aggregation_from_parsed.py` - Aggregation logic

---

### ✅ Step 4: Output Files
**Status**: ✅ **COMPLETE**

**Single student output:**
- ✅ `output/skill_profile_{student_id}.csv` - Per-student file
- ✅ `output/skill_profile_parsed_single.csv` - Single student format (overwritten each time)

**Batch output (for batch processing):**
- ✅ `output/skill_profiles_explainable.csv` - Multiple students

**Output columns:**
```
StudentID, Skill, EvidenceCount, ScoreNormalized, SkillLevel
```

**Files:**
- `backend/src/api/main.py` - Saves both formats after upload
- `backend/src/skill_aggregation_explainable.py` - Batch processing

---

### ✅ Step 5: Frontend Display
**Status**: ✅ **COMPLETE**

**"Skill Profile (from transcript)" screen shows:**

1. ✅ **List of skills** with:
   - Skill name
   - Score (normalized 0-1)
   - Evidence count (number of courses)
   - Level label (Beginner/Developing/Proficient/Advanced)

2. ✅ **Visual elements**:
   - Color-coded level badges
   - Progress bars for scores
   - Skill categorization (Technical/Soft Skills/Domain Knowledge)
   - Grid/card layout

3. ✅ **Filtering/Selection**:
   - Select skills for quiz (up to 5)
   - Filter by category
   - Search functionality

**Files:**
- `frontend-react/src/components/SkillsPage.jsx` - Main skill display
- `frontend-react/src/components/SkillTable.jsx` - Table view
- `frontend-react/src/components/TranscriptDisplay.jsx` - Skills preview

---

## 📊 Score Calculation Details

### Year Weights
```python
YEAR_WEIGHTS = {
    1: 0.8,   # Foundation courses
    2: 1.0,   # Core courses (baseline)
    3: 1.1,   # Advanced courses
    4: 1.2,   # Specialization courses (highest)
}
```

**Rationale**: Final year courses (Year 4) are more specialized and indicate deeper knowledge, so they get higher weight.

### Grade Points
```python
GRADE_POINTS = {
    "A+": 4.0, "A": 4.0, "A-": 3.7,
    "B+": 3.3, "B": 3.0, "B-": 2.7,
    "C+": 2.3, "C": 2.0, "C-": 1.7,
    "D+": 1.3, "D": 1.0, "D-": 0.7,
    "E": 0.0, "F": 0.0,
}
```

### Normalization
- **Max possible score**: 4.0 (A+) × 1.2 (Year 4) = 4.8
- **Normalized score**: `Average Contribution / 4.8`
- **Clipped to**: [0, 1]

### Skill Levels
- **Advanced**: Score ≥ 0.75
- **Proficient**: Score ≥ 0.50
- **Developing**: Score ≥ 0.25
- **Beginner**: Score < 0.25

---

## 🔄 Data Flow

```
1. Courses parsed from transcript
   ↓
2. For each course:
   - Find in course_skill_mapping.csv
   - Get skills list
   - Calculate: GradePoint × YearWeight = Contribution
   - Distribute contribution across skills
   ↓
3. Aggregate per skill:
   - Sum contributions
   - Count evidence (number of courses)
   - Normalize score (0-1)
   - Assign level
   ↓
4. Save to:
   - skill_profile_{student_id}.csv
   - skill_profile_parsed_single.csv
   ↓
5. Frontend displays:
   - Skills list
   - Scores & levels
   - Evidence counts
```

---

## 📁 Output File Structure

### `output/skill_profile_{student_id}.csv`
```csv
StudentID,Skill,EvidenceCount,ScoreNormalized,SkillLevel
IT21013928,SQL & Relational Databases,3,0.85,Advanced
IT21013928,Python Programming,2,0.72,Proficient
IT21013928,Data Structures & Algorithms,1,0.65,Proficient
```

### `output/skill_profile_parsed_single.csv`
Same format (single student, overwritten each upload)

### `output/skill_profiles_explainable.csv`
Batch format (multiple students)

---

## 🎨 Frontend Display Features

### Skill Cards Display:
- ✅ Skill name
- ✅ Progress bar (visual score indicator)
- ✅ Score percentage (0-100%)
- ✅ Level badge (color-coded)
- ✅ Evidence count (number of courses)
- ✅ Category label (Technical/Soft Skills/Domain)

### Interactive Features:
- ✅ Click to select/deselect skills
- ✅ Filter by category
- ✅ Search by skill name
- ✅ Sort by score/level/evidence

---

## ✅ Verification Checklist

- [x] Course → Skill mapping loaded correctly
- [x] Year weights applied (Year 4 = 1.2, Year 1 = 0.8)
- [x] Grade points calculated correctly
- [x] Contributions aggregated per skill
- [x] Scores normalized to [0, 1]
- [x] Skill levels assigned (Beginner/Developing/Proficient/Advanced)
- [x] Evidence count tracked
- [x] Saved to `skill_profile_parsed_single.csv`
- [x] Frontend displays skills with levels
- [x] Frontend shows evidence counts

---

## 📝 Example Calculation

**Student: IT21013928**

**Courses:**
- IT2100 (Database Systems), Year 2, Grade: A
- IT4100 (Advanced ML), Year 4, Grade: B+
- IT3100 (Data Mining), Year 3, Grade: A

**Skill: "Machine Learning"**

**Contributions:**
- IT4100: 3.3 (B+) × 1.2 (Year 4) = 3.96
- IT3100: 4.0 (A) × 1.1 (Year 3) = 4.40
- Total: 8.36
- EvidenceCount: 2
- Average: 4.18
- ScoreNormalized: 4.18 / 4.8 = 0.87
- SkillLevel: **Advanced** (≥ 0.75)

---

## ⚠️ Manual Setup Notes

**None required!** Skill inference works automatically.

**Optional customization:**
- Adjust year weights in `skill_aggregation_from_parsed.py` if needed
- Modify skill level thresholds if needed
- Maintain `course_skill_mapping.csv` with course → skill mappings

---

## 🎯 Summary

**Status**: ✅ **COMPLETE**

All requirements for Step 3 (Skill Inference) are implemented:
- ✅ Year-weighted scoring (final year courses weighted more)
- ✅ Grade point calculation
- ✅ Course importance (via year weights)
- ✅ Aggregation per student
- ✅ Output to `skill_profile_parsed_single.csv`
- ✅ Frontend display with levels and evidence counts

**Enhancement added**: Year weights now properly applied in the API endpoint (previously only in batch processing).

