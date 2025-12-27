# src/api/main.py
from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    Form,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

import os
import io
import pandas as pd

# Import your existing helpers
from src.course_skill_mapping import load_course_skill_mapping
from src import transcript_ingestion  # uses extract_text_from_file, parse_transcript_text
from src.transcript_ingestion import parse_transcript_file
from src.skill_aggregation_from_parsed import build_skill_profile_from_parsed

app = FastAPI(
    title="Transcript-based Skill Validation API",
    version="0.3.0",
    description="Backend API for transcript → skills → quiz → role alignment.",
)

# ---------------------------------------------------------------------
# CORS (frontend at http://127.0.0.1:5500 or http://localhost:5500)
# ---------------------------------------------------------------------
origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------
# Utility: core data loading
# ---------------------------------------------------------------------
def load_skill_profiles() -> pd.DataFrame:
    """
    Prefer fused skill profiles (transcript + quiz).
    Fallback to explainable baseline if fused does not exist.
    """
    candidates = [
        "output/skill_profiles_with_quiz.csv",
        "output/skill_profiles_explainable.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            return pd.read_csv(path)
    raise FileNotFoundError(
        "No skill profile file found in output/. Run aggregation pipeline first."
    )


def load_role_readiness() -> pd.DataFrame:
    """
    Static role readiness from the offline dynamic role model
    (job_role_model_dynamic.py).
    """
    path = "output/role_readiness_dynamic.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Role readiness file not found. Run job_role_model_dynamic.py first."
        )
    return pd.read_csv(path)

BASELINE_SKILL_FILE = "output/skill_profiles_explainable.csv"
FUSED_SKILL_FILE = "output/skill_profiles_with_quiz.csv"
ROLE_TEMPLATE_FILE = "output/job_role_skill_templates_dynamic.csv"


def load_baseline_skill_profiles() -> pd.DataFrame:
    """
    Load the grade-based skill profiles (without quiz fusion).
    Used as the stable baseline when fusing quiz results.
    """
    if not os.path.exists(BASELINE_SKILL_FILE):
        raise FileNotFoundError(
            f"Baseline skill profile file not found: {BASELINE_SKILL_FILE}. "
            f"Run skill_aggregation_explainable.py first."
        )
    return pd.read_csv(BASELINE_SKILL_FILE)



def load_role_templates() -> pd.DataFrame:
    """
    Role → skill template matrix derived from real job postings.
    Used for *updated* role matches after quiz fusion.
    """
    path = "output/job_role_skill_templates_dynamic.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Role-skill template file not found. Run job_postings_ingestion.py first."
        )
    df = pd.read_csv(path)
    if "ImportanceNorm" not in df.columns:
        df["ImportanceNorm"] = 1.0
    return df


def load_quiz_questions() -> pd.DataFrame:
    path = "output/quiz_questions_generated.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Quiz questions file not found. Run quiz_generation_rag.py first."
        )
    df = pd.read_csv(path)

    # Ensure QuestionID exists
    if "QuestionID" not in df.columns:
        df = df.reset_index().rename(columns={"index": "QuestionID"})
    return df


# ---------------------------------------------------------------------
# Utility: grade → numeric + transcript-based skill aggregation
# ---------------------------------------------------------------------
GRADE_TO_POINT = {
    "A+": 4.0,
    "A": 4.0,
    "A-": 3.7,
    "B+": 3.3,
    "B": 3.0,
    "B-": 2.7,
    "C+": 2.3,
    "C": 2.0,
    "C-": 1.7,
    "D+": 1.3,
    "D": 1.0,
    "E": 0.0,
    "F": 0.0,
}

YEAR_WEIGHTS = {
    1: 0.8,
    2: 1.0,
    3: 1.2,
    4: 1.5,
}


def grade_to_normalized(grade: str) -> float:
    g = (grade or "").strip().upper()
    if g not in GRADE_TO_POINT:
        return 0.0
    return GRADE_TO_POINT[g] / 4.0  # scale into [0,1]


def aggregate_skills_from_parsed(df_parsed: pd.DataFrame) -> pd.DataFrame:
    """
    Given a parsed transcript (one student), join with course-skill mapping
    and compute a transcript-based skill profile.

    Returns columns:
      StudentID, Skill, EvidenceCount, TotalContribution, ScoreNormalized, SkillLevel
    """
    if df_parsed.empty:
        return pd.DataFrame()

    mapping = load_course_skill_mapping("input/course_skill_mapping.csv")

    records = []
    for _, row in df_parsed.iterrows():
        course_code = str(row.get("CourseCode", "")).strip()
        grade = str(row.get("Grade", "")).strip()
        if not course_code or course_code not in mapping or not grade:
            continue

        student_id = str(row.get("StudentID", "")).strip()
        if not student_id:
            continue

        year_val = row.get("Year", None)
        try:
            year_int = int(year_val) if pd.notna(year_val) else None
        except Exception:
            year_int = None

        year_weight = YEAR_WEIGHTS.get(year_int, 1.0)
        grade_norm = grade_to_normalized(grade)

        course_info = mapping[course_code]
        skills = course_info.get("skills", [])

        for sk in skills:
            records.append(
                {
                    "StudentID": student_id,
                    "Skill": sk,
                    "GradeNorm": grade_norm,
                    "Year": year_int,
                    "YearWeight": year_weight,
                    "Contribution": grade_norm * year_weight,
                }
            )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    grouped = (
        df.groupby(["StudentID", "Skill"], as_index=False)
        .agg(
            EvidenceCount=("Skill", "size"),
            TotalContribution=("Contribution", "sum"),
        )
    )

    def _normalize_group(g: pd.DataFrame) -> pd.DataFrame:
        max_c = g["TotalContribution"].max()
        if max_c <= 0:
            g["ScoreNormalized"] = 0.0
        else:
            g["ScoreNormalized"] = g["TotalContribution"] / max_c
        return g

    grouped = grouped.groupby("StudentID", group_keys=False).apply(_normalize_group)

    levels = []
    for s in grouped["ScoreNormalized"]:
        if s >= 0.66:
            levels.append("Advanced")
        elif s >= 0.33:
            levels.append("Developing")
        else:
            levels.append("Beginner")
    grouped["SkillLevel"] = levels

    return grouped


# ---------------------------------------------------------------------
# Utility: helpers for updated role matches after quiz
# ---------------------------------------------------------------------
def compute_roles_for_student_from_skills(
    student_skills: pd.DataFrame,
    templates_df: pd.DataFrame,
    score_col: str = "FusedScore",
    weak_threshold: float = 0.4,
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """
    Compute role readiness for a single student given a skill profile
    and role-skill templates.
    """
    if student_skills.empty or templates_df.empty:
        return []

    # ensure we only keep skill + score
    s_df = student_skills[["Skill", score_col]].copy()
    s_df = s_df.rename(columns={score_col: "Score"})
    roles = []

    for role_name in templates_df["RoleName"].unique():
        role_skills = templates_df[templates_df["RoleName"] == role_name]
        if role_skills.empty:
            continue

        total_importance = role_skills["ImportanceNorm"].sum()
        if total_importance <= 0:
            continue

        attained_weighted = 0.0
        num_skills = len(role_skills)
        num_present = 0
        weak_or_missing: List[str] = []

        for _, r in role_skills.iterrows():
            skill = r["Skill"]
            importance = float(r["ImportanceNorm"])
            row = s_df[s_df["Skill"] == skill]

            if row.empty:
                student_score = 0.0
            else:
                student_score = float(row.iloc[0]["Score"])
                num_present += 1

            attained_weighted += importance * student_score
            if student_score < weak_threshold:
                weak_or_missing.append(skill)

        readiness = attained_weighted / total_importance
        coverage = num_present / num_skills if num_skills else 0.0

        roles.append(
            {
                "role_name": role_name,
                "readiness_score": readiness,
                "coverage": coverage,
                "num_skills": num_skills,
                "num_skills_present": num_present,
                "num_weak_or_missing": len(weak_or_missing),
                "weak_or_missing_skills": ", ".join(weak_or_missing[:15]),
            }
        )

    roles_sorted = sorted(roles, key=lambda r: r["readiness_score"], reverse=True)
    return roles_sorted[:top_n]


# ---------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------
class QuizResponse(BaseModel):
    question_id: int
    selected_option: str
    response_time_seconds: Optional[float] = None


class QuizSubmission(BaseModel):
    responses: List[QuizResponse]


# ---------------------------------------------------------------------
# Endpoints: main dataset / fused profile
# ---------------------------------------------------------------------
@app.get("/students/{student_id}/skills")
def get_student_skills(student_id: str, top_n: int = 15) -> Dict[str, Any]:
    """
    Skill profile from the main dataset / fused pipeline
    (historical transcripts + quiz fusion).
    """
    try:
        df = load_skill_profiles()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    df = df[df["StudentID"] == student_id]
    if df.empty:
        raise HTTPException(
            status_code=404, detail=f"No skill profile found for student {student_id}"
        )

    score_col = "FinalScore" if "FinalScore" in df.columns else "ScoreNormalized"
    level_col = "FinalSkillLevel" if "FinalSkillLevel" in df.columns else "SkillLevel"

    df = df.sort_values(score_col, ascending=False).head(top_n)

    skills = []
    for _, row in df.iterrows():
        try:
            score_val = float(row[score_col])
        except Exception:
            score_val = 0.0

        skills.append(
            {
                "skill": str(row["Skill"]),
                "score": score_val,
                "level": str(row[level_col]),
                "evidence_count": int(row.get("EvidenceCount", 0)),
            }
        )

    return {
        "student_id": student_id,
        "count": len(skills),
        "skills": skills,
    }


@app.get("/students/{student_id}/roles")
def get_student_roles(student_id: str, top_n: int = 10) -> Dict[str, Any]:
    """
    Baseline role matches from the offline dynamic role model.
    These are based on the main dataset (fused skill profiles).
    """
    try:
        df = load_role_readiness()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    df = df[df["StudentID"] == student_id]
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No role readiness found for student {student_id}. "
            f"Check that the offline pipeline has been run.",
        )

    df = df.sort_values("ReadinessScore", ascending=False).head(top_n)

    roles = []
    for _, row in df.iterrows():
        roles.append(
            {
                "role_name": str(row["RoleName"]),
                "readiness_score": float(row.get("ReadinessScore", 0.0)),
                "coverage": float(row.get("Coverage", 0.0)),
                "num_skills": int(row.get("NumSkills", 0)),
                "num_skills_present": int(row.get("NumSkillsPresent", 0)),
                "num_weak_or_missing": int(row.get("NumWeakOrMissing", 0)),
                "weak_or_missing_skills": str(row.get("WeakOrMissingSkills", "")),
            }
        )

    return {
        "student_id": student_id,
        "count": len(roles),
        "roles": roles,
    }


# ---------------------------------------------------------------------
# Endpoint: upload transcript → transcript-based skill profile
# ---------------------------------------------------------------------
@app.post("/students/{student_id}/upload-transcript")
async def upload_transcript_for_student(
    student_id: str,
    file: UploadFile = File(...),
    regno: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """
    1) Save uploaded transcript (PDF/image).
    2) Parse it into courses/grades.
    3) Map courses → skills.
    4) Save skill profile for this student and return it.

    This does NOT touch the big dataset; it's a per-student profile:
      output/skill_profile_parsed_{student_id}.csv
    """
    # 1) Save uploaded file
    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)

    # Make a simple safe filename
    original_name = os.path.basename(file.filename)
    saved_path = os.path.join(uploads_dir, f"{student_id}_{original_name}")

    with open(saved_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # 2) Parse transcript into a course-level DataFrame
    try:
        parsed_df = parse_transcript_file(
            file_path=saved_path,
            student_id=student_id,
            regno=regno or student_id,
            tesseract_cmd=None,  # set path here if you need
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse transcript: {e}")

    if parsed_df.empty:
        raise HTTPException(status_code=400, detail="Parsed transcript is empty. Check the file format/content.")

    # 3) Build skill profile from parsed transcript
    try:
        skill_df = build_skill_profile_from_parsed(parsed_df, mapping_path="input/course_skill_mapping.csv")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build skill profile: {e}")

    if skill_df.empty:
        raise HTTPException(
            status_code=400,
            detail="No skills could be derived from this transcript (no matching course codes in mapping).",
        )

    # 4) Save per-student skill profile
    out_path = f"output/skill_profile_parsed_{student_id}.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    skill_df.to_csv(out_path, index=False)

    # 5) Return skills in the same shape as /students/{id}/skills
    skills_payload = []
    for _, row in skill_df.sort_values("ScoreNormalized", ascending=False).iterrows():
        skills_payload.append(
            {
                "skill": row["Skill"],
                "score": float(row["ScoreNormalized"]),
                "level": str(row["SkillLevel"]),
                "evidence_count": int(row.get("EvidenceCount", 0)),
            }
        )

    return {
        "student_id": student_id,
        "from": "uploaded_transcript",
        "count": len(skills_payload),
        "skills": skills_payload,
        "parsed_courses": len(parsed_df),
        "saved_profile_path": out_path,
    }



# ---------------------------------------------------------------------
# Endpoint: prepare quiz (questions still come from question bank)
# ---------------------------------------------------------------------
@app.post("/students/{student_id}/prepare-quiz")
def prepare_quiz(student_id: str, max_questions: int = 5) -> Dict[str, Any]:
    """
    Return a small quiz for the student by sampling questions from
    output/quiz_questions_generated.csv.

    At the moment we filter by StudentID if that column exists;
    otherwise we just sample globally.
    """
    try:
        df = load_quiz_questions()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if "StudentID" in df.columns:
        df = df[df["StudentID"] == student_id]

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No questions found for student {student_id}. "
                "Run quiz_generation_rag.py or relax filtering."
            ),
        )

    if len(df) > max_questions:
        df = df.sample(n=max_questions, random_state=42)

    questions = []
    for _, row in df.iterrows():
        qid = int(row["QuestionID"])
        options = {
            "A": str(row.get("OptionA", "")),
            "B": str(row.get("OptionB", "")),
            "C": str(row.get("OptionC", "")),
            "D": str(row.get("OptionD", "")),
        }
        questions.append(
            {
                "question_id": qid,
                "question_text": str(row.get("QuestionText", "")),
                "skill": str(row.get("Skill", "")),
                "difficulty": str(row.get("Difficulty", "Unknown")),
                "role_name": str(row.get("RoleName", "")),
                "options": options,
            }
        )

    return {
        "student_id": student_id,
        "num_questions": len(questions),
        "questions": questions,
    }


# ---------------------------------------------------------------------
# Endpoint: submit quiz → score → updated roles (in-memory)
# ---------------------------------------------------------------------
@app.post("/students/{student_id}/submit-quiz")
def submit_quiz(student_id: str, submission: QuizSubmission) -> Dict[str, Any]:
    """
    1) Score quiz answers in memory.
    2) Fuse quiz results into the student's skill profile
       (creating/updating output/skill_profiles_with_quiz.csv).
    3) Recompute role matches for this student from the fused skills.
    """
    if not submission.responses:
        raise HTTPException(status_code=400, detail="No responses submitted.")

    try:
        questions_df = load_quiz_questions()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Step 1: pure scoring
    try:
        score_info = _score_quiz_in_memory(student_id, submission, questions_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scoring quiz: {e}")

    # Default outputs in case later steps fail
    updated_skills: List[Dict[str, Any]] = []
    updated_roles: List[Dict[str, Any]] = []
    note_parts = ["Quiz scored successfully."]

    # Step 2: fuse into skill profile
    try:
        fused_student = _fuse_skills_with_quiz(student_id, score_info["per_skill"])
        if not fused_student.empty:
            score_col = "FinalScore" if "FinalScore" in fused_student.columns else "ScoreNormalized"
            level_col = "FinalSkillLevel" if "FinalSkillLevel" in fused_student.columns else "SkillLevel"

            # return top 15 updated skills
            for _, row in fused_student.sort_values(score_col, ascending=False).head(15).iterrows():
                updated_skills.append(
                    {
                        "skill": str(row["Skill"]),
                        "score": float(row[score_col]),
                        "level": str(level_col in row and row[level_col] or row.get("SkillLevel", "")),
                    }
                )
            note_parts.append("Skill profile fused with quiz results.")
        else:
            note_parts.append("Student not found in baseline skill file; fusion skipped.")
    except FileNotFoundError as e:
        note_parts.append(f"Fusion skipped: {e}")
    except Exception as e:
        note_parts.append(f"Fusion error: {e}")

    # Step 3: recompute roles
    try:
        updated_roles = _recompute_roles_for_student(student_id, max_roles=5)
        if updated_roles:
            note_parts.append("Role matches recomputed from fused skills.")
        else:
            note_parts.append("No updated roles available (missing templates or skills).")
    except Exception as e:
        note_parts.append(f"Role recomputation error: {e}")

    return {
        "student_id": student_id,
        "num_answered": score_info["num_answered"],
        "num_correct": score_info["num_correct"],
        "overall_accuracy": score_info["overall_accuracy"],
        "per_skill": score_info["per_skill"],
        "detailed": score_info["detailed"],
        "updated_skills": updated_skills,
        "updated_roles": updated_roles,
        "note": " ".join(note_parts),
    }


def _score_quiz_in_memory(
    student_id: str,
    submission: QuizSubmission,
    questions_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Pure scoring: compare selected vs correct answer using quiz_questions_generated.csv.
    Returns per-question and per-skill stats (no CSV writes).
    """
    if questions_df.empty:
        raise ValueError("Question bank is empty.")

    # Ensure QuestionID is an index
    if "QuestionID" not in questions_df.columns:
        questions_df = questions_df.reset_index().rename(columns={"index": "QuestionID"})
    questions_df = questions_df.set_index("QuestionID")

    detailed_results = []
    correct_count = 0

    # per-skill aggregation
    skill_stats: Dict[str, Dict[str, float]] = {}
    difficulty_weight = {"Easy": 0.3, "Medium": 0.6, "Hard": 1.0}

    for r in submission.responses:
        qid = r.question_id
        if qid not in questions_df.index:
            # ignore unknown or stale question ids
            continue

        row = questions_df.loc[qid]
        correct = str(row.get("CorrectOption", "")).strip().upper()
        selected = r.selected_option.strip().upper()
        skill = str(row.get("Skill", "Unknown"))
        difficulty = str(row.get("Difficulty", "Unknown"))
        role_name = str(row.get("RoleName", ""))

        is_correct = selected == correct
        if is_correct:
            correct_count += 1

        detailed_results.append(
            {
                "question_id": qid,
                "skill": skill,
                "selected_option": selected,
                "correct_option": correct,
                "is_correct": is_correct,
                "difficulty": difficulty,
                "role_name": role_name,
                "response_time_seconds": r.response_time_seconds,
            }
        )

        # aggregate per skill
        if skill not in skill_stats:
            skill_stats[skill] = {
                "num": 0,
                "correct": 0,
                "sum_diff": 0.0,
                "sum_time": 0.0,
            }
        skill_stats[skill]["num"] += 1
        if is_correct:
            skill_stats[skill]["correct"] += 1

        diff_num = difficulty_weight.get(difficulty, 0.5)
        skill_stats[skill]["sum_diff"] += diff_num
        if r.response_time_seconds is not None:
            skill_stats[skill]["sum_time"] += float(r.response_time_seconds)

    total_answered = sum(s["num"] for s in skill_stats.values())
    overall_accuracy = correct_count / total_answered if total_answered > 0 else 0.0

    per_skill = []
    for skill, stats in skill_stats.items():
        n = stats["num"]
        acc = stats["correct"] / n if n else 0.0
        avg_diff = stats["sum_diff"] / n if n else 0.0
        avg_time = stats["sum_time"] / n if n else None
        per_skill.append(
            {
                "skill": skill,
                "num_questions": n,
                "num_correct": stats["correct"],
                "accuracy": acc,
                "avg_difficulty_numeric": avg_diff,
                "avg_response_time": avg_time,
            }
        )

    return {
        "student_id": student_id,
        "num_answered": total_answered,
        "num_correct": correct_count,
        "overall_accuracy": overall_accuracy,
        "per_skill": per_skill,
        "detailed": detailed_results,
    }

def _score_to_level(score: float) -> str:
    """
    Map final numeric score [0,1] into a human-readable skill level.
    Tune thresholds as you like for your thesis.
    """
    if score >= 0.75:
        return "Advanced"
    elif score >= 0.5:
        return "Developing"
    else:
        return "Beginner"


def _fuse_skills_with_quiz(
    student_id: str,
    per_skill_stats: List[Dict[str, Any]],
    baseline_weight: float = 0.7,
    quiz_weight: float = 0.3,
) -> pd.DataFrame:
    """
    Take baseline skill profile + quiz accuracies and produce
    output/skill_profiles_with_quiz.csv, returning this student's rows.
    """
    # 1) Load baseline skill profiles (transcript-based)
    baseline = load_baseline_skill_profiles()

    # sanity check
    if "ScoreNormalized" not in baseline.columns:
        raise RuntimeError("Baseline skill file must have 'ScoreNormalized' column.")

    # 2) Copy to fused, so we keep all other students untouched
    fused = baseline.copy()

    # Index quiz stats by skill for faster lookup
    quiz_by_skill = {row["skill"]: row for row in per_skill_stats}

    # 3) For this student, adjust skills that appeared in the quiz
    mask_student = fused["StudentID"] == student_id
    student_rows = fused[mask_student].copy()
    if student_rows.empty:
        # If student is not in baseline (e.g., only from parsed transcript), you can
        # extend this to append new rows. For now, we just skip fusion.
        return student_rows

    # ensure FinalScore column exists
    if "FinalScore" not in fused.columns:
        fused["FinalScore"] = fused["ScoreNormalized"]

    for idx, row in student_rows.iterrows():
        skill_name = row["Skill"]
        if skill_name not in quiz_by_skill:
            # no quiz evidence for this skill, keep baseline
            continue

        baseline_score = float(row["ScoreNormalized"])
        quiz_accuracy = float(quiz_by_skill[skill_name]["accuracy"])

        final_score = (
            baseline_weight * baseline_score + quiz_weight * quiz_accuracy
        )

        fused.loc[idx, "FinalScore"] = final_score

    # 4) Fill any NaNs and assign FinalSkillLevel
    fused["FinalScore"] = fused["FinalScore"].fillna(fused["ScoreNormalized"])
    fused["FinalSkillLevel"] = fused["FinalScore"].apply(_score_to_level)

    # 5) Persist for other endpoints (/students/{id}/skills, etc.)
    fused.to_csv(FUSED_SKILL_FILE, index=False)

    # return this student's fused rows
    fused_student = fused[fused["StudentID"] == student_id].copy()
    return fused_student

def _score_to_level(score: float) -> str:
    """
    Map final numeric score [0,1] into a human-readable skill level.
    Tune thresholds as you like for your thesis.
    """
    if score >= 0.75:
        return "Advanced"
    elif score >= 0.5:
        return "Developing"
    else:
        return "Beginner"


def _fuse_skills_with_quiz(
    student_id: str,
    per_skill_stats: List[Dict[str, Any]],
    baseline_weight: float = 0.7,
    quiz_weight: float = 0.3,
) -> pd.DataFrame:
    """
    Take baseline skill profile + quiz accuracies and produce
    output/skill_profiles_with_quiz.csv, returning this student's rows.
    """
    # 1) Load baseline skill profiles (transcript-based)
    baseline = load_baseline_skill_profiles()

    # sanity check
    if "ScoreNormalized" not in baseline.columns:
        raise RuntimeError("Baseline skill file must have 'ScoreNormalized' column.")

    # 2) Copy to fused, so we keep all other students untouched
    fused = baseline.copy()

    # Index quiz stats by skill for faster lookup
    quiz_by_skill = {row["skill"]: row for row in per_skill_stats}

    # 3) For this student, adjust skills that appeared in the quiz
    mask_student = fused["StudentID"] == student_id
    student_rows = fused[mask_student].copy()
    if student_rows.empty:
        # If student is not in baseline (e.g., only from parsed transcript), you can
        # extend this to append new rows. For now, we just skip fusion.
        return student_rows

    # ensure FinalScore column exists
    if "FinalScore" not in fused.columns:
        fused["FinalScore"] = fused["ScoreNormalized"]

    for idx, row in student_rows.iterrows():
        skill_name = row["Skill"]
        if skill_name not in quiz_by_skill:
            # no quiz evidence for this skill, keep baseline
            continue

        baseline_score = float(row["ScoreNormalized"])
        quiz_accuracy = float(quiz_by_skill[skill_name]["accuracy"])

        final_score = (
            baseline_weight * baseline_score + quiz_weight * quiz_accuracy
        )

        fused.loc[idx, "FinalScore"] = final_score

    # 4) Fill any NaNs and assign FinalSkillLevel
    fused["FinalScore"] = fused["FinalScore"].fillna(fused["ScoreNormalized"])
    fused["FinalSkillLevel"] = fused["FinalScore"].apply(_score_to_level)

    # 5) Persist for other endpoints (/students/{id}/skills, etc.)
    fused.to_csv(FUSED_SKILL_FILE, index=False)

    # return this student's fused rows
    fused_student = fused[fused["StudentID"] == student_id].copy()
    return fused_student
