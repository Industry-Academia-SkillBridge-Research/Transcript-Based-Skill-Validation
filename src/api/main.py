# src/api/main.py

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple 

import os
import sys
import subprocess
import pandas as pd

# Base paths for calling CLI scripts and saving uploads
API_DIR = os.path.dirname(os.path.abspath(__file__))      # .../src/api
SRC_DIR = os.path.dirname(API_DIR)                        # .../src
UPLOAD_DIR = os.path.join(SRC_DIR, "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)

TRANSCRIPT_SCRIPT = os.path.join(SRC_DIR, "transcript_ingestion.py")
AGGREGATION_SCRIPT = os.path.join(SRC_DIR, "skill_aggregation_from_parsed.py")


app = FastAPI(
    title="Transcript-based Skill Validation API",
    version="0.3.0",
    description="Backend API for transcript → skills → roles → quiz pipeline."
)

# CORS so your static frontend (http://127.0.0.1:5500) can call the API
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

# ---------- Utility loaders ----------


def load_skill_profiles() -> pd.DataFrame:
    """
    Prefer fused skill profiles (after quiz), otherwise baseline explainable profiles.
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
    path = "output/role_readiness_dynamic.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Role readiness file not found. Run job_role_model_dynamic.py first."
        )
    return pd.read_csv(path)

def load_role_templates_dynamic() -> pd.DataFrame:
    """
    Role–skill templates built from real job postings
    (output/job_role_skill_templates_dynamic.csv).
    """
    path = "output/job_role_skill_templates_dynamic.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Role templates file not found. Run job_postings_ingestion.py first."
        )
    return pd.read_csv(path)


def compute_role_readiness_from_profile(
    skill_df: pd.DataFrame,
    templates_df: pd.DataFrame,
    weak_threshold: float = 0.4,
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """
    Compute role readiness for ONE student, given a skill profile DataFrame
    (columns: Skill, ScoreNormalized, SkillLevel) and the dynamic templates.
    This version does NOT write any CSV; it just returns a list of roles.
    """
    if skill_df.empty or templates_df.empty:
        return []

    if "Skill" not in skill_df.columns or "ScoreNormalized" not in skill_df.columns:
        raise ValueError("Skill profile must have 'Skill' and 'ScoreNormalized' columns.")

    profiles_indexed = skill_df.set_index("Skill")

    roles: List[Dict[str, Any]] = []

    for role_name, role_skills in templates_df.groupby("RoleName"):
        total_importance = role_skills["ImportanceNorm"].sum()
        if total_importance == 0:
            continue

        attained_weighted = 0.0
        num_skills = len(role_skills)
        num_present = 0
        weak_or_missing: List[str] = []

        for _, r in role_skills.iterrows():
            skill = r["Skill"]
            importance = float(r["ImportanceNorm"])

            student_score = 0.0
            if skill in profiles_indexed.index:
                student_score = float(profiles_indexed.loc[skill, "ScoreNormalized"])
                if student_score > 0:
                    num_present += 1

            attained_weighted += importance * student_score

            if student_score < weak_threshold:
                weak_or_missing.append(skill)

        if total_importance == 0:
            continue

        readiness_score = attained_weighted / total_importance
        coverage = num_present / num_skills if num_skills else 0.0

        roles.append(
            {
                "role_name": role_name,
                "readiness_score": readiness_score,
                "coverage": coverage,
                "num_skills": num_skills,
                "num_skills_present": num_present,
                "num_weak_or_missing": len(weak_or_missing),
                "weak_or_missing_skills": ", ".join(weak_or_missing[:15]),
            }
        )

    roles_sorted = sorted(roles, key=lambda r: r["readiness_score"], reverse=True)
    return roles_sorted[:top_n]


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


def load_role_templates() -> pd.DataFrame:
    """
    Role–skill templates built from real job postings.
    """
    path = "output/job_role_skill_templates_dynamic.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Role templates file not found. Run job_postings_ingestion.py first."
        )
    return pd.read_csv(path)


# ---------- Pydantic models ----------

class QuizResponse(BaseModel):
    question_id: int
    selected_option: str
    response_time_seconds: Optional[float] = None


class QuizSubmission(BaseModel):
    responses: List[QuizResponse]


# ---------- Helper: fuse quiz into skill profiles ----------

def fuse_quiz_into_skill_profiles(
    student_id: str,
    per_skill_stats: List[Dict[str, Any]],
    quiz_weight: float = 0.5,
) -> pd.DataFrame:
    """
    Update skill_profiles_with_quiz.csv (or create it from explainable profiles)
    by injecting quiz accuracy for the given student and skills.

    Returns the updated DataFrame.
    """
    # Load baseline / fused profiles
    base_path = None
    if os.path.exists("output/skill_profiles_with_quiz.csv"):
        base_path = "output/skill_profiles_with_quiz.csv"
    elif os.path.exists("output/skill_profiles_explainable.csv"):
        base_path = "output/skill_profiles_explainable.csv"
    else:
        raise FileNotFoundError(
            "No skill profiles found to fuse quiz into. "
            "Expected output/skill_profiles_with_quiz.csv or "
            "output/skill_profiles_explainable.csv."
        )

    df = pd.read_csv(base_path)

    # We assume baseline score is in ScoreNormalized
    if "ScoreNormalized" not in df.columns:
        raise ValueError("Expected 'ScoreNormalized' column in skill profiles.")

    # Ensure quiz-related columns exist
    if "QuizProficiency" not in df.columns:
        df["QuizProficiency"] = 0.0
    if "QuizProficiencyFilled" not in df.columns:
        df["QuizProficiencyFilled"] = 0.0

    # Inject quiz accuracy for this student + skill
    for s in per_skill_stats:
        skill = s["skill"]
        acc = float(s["accuracy"])
        mask = (df["StudentID"] == student_id) & (df["Skill"] == skill)
        if mask.any():
            df.loc[mask, "QuizProficiency"] = acc

    # Fill QuizProficiencyFilled:
    # if QuizProficiency > 0 → use it, else fallback to ScoreNormalized
    df["QuizProficiencyFilled"] = df["QuizProficiency"]
    need_fallback = df["QuizProficiencyFilled"] <= 0
    df.loc[need_fallback, "QuizProficiencyFilled"] = df.loc[
        need_fallback, "ScoreNormalized"
    ]

    # Compute fused final score
    alpha = float(quiz_weight)
    df["FinalScore"] = (
        (1.0 - alpha) * df["ScoreNormalized"]
        + alpha * df["QuizProficiencyFilled"]
    )

    # Map FinalScore to levels
    def score_to_level(x: float) -> str:
        if x < 0.25:
            return "Beginner"
        elif x < 0.5:
            return "Developing"
        elif x < 0.75:
            return "Proficient"
        else:
            return "Advanced"

    df["FinalSkillLevel"] = df["FinalScore"].apply(score_to_level)

    # Save fused file back
    df.to_csv("output/skill_profiles_with_quiz.csv", index=False)
    return df


# ---------- Helper: recompute role readiness for one student ----------

def recompute_role_readiness_for_student(
    student_id: str,
    profiles_df: Optional[pd.DataFrame] = None,
    weak_threshold: float = 0.4,
) -> pd.DataFrame:
    """
    Recompute dynamic role readiness for a single student using
    job_role_skill_templates_dynamic.csv and save back into
    output/role_readiness_dynamic.csv.

    Returns the new readiness summary for that student.
    """
    if profiles_df is None:
        profiles_df = load_skill_profiles()

    templates_df = load_role_templates()

    # Score column: prefer FinalScore if present
    score_col = "FinalScore" if "FinalScore" in profiles_df.columns else "ScoreNormalized"
    level_col = "FinalSkillLevel" if "FinalSkillLevel" in profiles_df.columns else "SkillLevel"

    student_profiles = profiles_df[profiles_df["StudentID"] == student_id].copy()
    if student_profiles.empty:
        return pd.DataFrame()

    # Index student skills
    profiles_indexed = student_profiles.set_index("Skill")

    roles = templates_df["RoleName"].unique()
    summary_records = []

    for role in roles:
        role_skills = templates_df[templates_df["RoleName"] == role]
        if role_skills.empty:
            continue

        total_importance = role_skills["ImportanceNorm"].sum()
        if total_importance == 0:
            continue

        attained_weighted = 0.0
        num_skills = len(role_skills)
        num_present = 0
        weak_or_missing = []

        for _, row in role_skills.iterrows():
            skill = row["Skill"]
            importance = float(row["ImportanceNorm"])

            student_score = 0.0
            if skill in profiles_indexed.index:
                s_row = profiles_indexed.loc[skill]
                student_score = float(s_row[score_col])
                num_present += 1

            attained_weighted += importance * student_score
            if student_score < weak_threshold:
                weak_or_missing.append(skill)

        readiness_score = attained_weighted / total_importance
        coverage = num_present / num_skills if num_skills > 0 else 0.0

        summary_records.append(
            {
                "StudentID": student_id,
                "RoleName": role,
                "ReadinessScore": readiness_score,
                "Coverage": coverage,
                "NumSkills": num_skills,
                "NumSkillsPresent": num_present,
                "NumWeakOrMissing": len(weak_or_missing),
                "WeakOrMissingSkills": ", ".join(weak_or_missing[:15]),
            }
        )

    new_summary = pd.DataFrame(summary_records)

    # Merge into global role_readiness_dynamic.csv
    path = "output/role_readiness_dynamic.csv"
    if os.path.exists(path):
        old = pd.read_csv(path)
        old = old[old["StudentID"] != student_id]
        merged = pd.concat([old, new_summary], ignore_index=True)
    else:
        merged = new_summary

    merged.to_csv(path, index=False)
    return new_summary


# ---------- Endpoints ----------

@app.get("/students/{student_id}/skills")
def get_student_skills(student_id: str, top_n: int = 15) -> Dict[str, Any]:
    """
    Return top skills for a student from the skill profile CSV.
    """
    try:
        df = load_skill_profiles()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    df = df[df["StudentID"] == student_id]
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No skill profile found for student {student_id}")

    score_col = "FinalScore" if "FinalScore" in df.columns else "ScoreNormalized"
    level_col = "FinalSkillLevel" if "FinalSkillLevel" in df.columns else "SkillLevel"

    df = df.sort_values(score_col, ascending=False).head(top_n)

    skills = [
        {
            "skill": row["Skill"],
            "score": float(row[score_col]),
            "level": str(row[level_col]),
            "evidence_count": int(row.get("EvidenceCount", 0)),
        }
        for _, row in df.iterrows()
    ]

    return {
        "student_id": student_id,
        "count": len(skills),
        "skills": skills,
    }

@app.post("/students/{student_id}/upload-transcript")
async def upload_transcript(
    student_id: str,
    file: UploadFile = File(...),
    regno: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """
    Upload a transcript PDF/image, run the existing CLI pipeline:
      - transcript_ingestion.py  → output/transcript_parsed_single.csv
      - skill_aggregation_from_parsed.py → output/skill_profile_parsed_single.csv

    Then compute skills + role matches just for this upload and return them
    (does not yet modify the big 1000-student CSVs).
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".pdf", ".png", ".jpg", ".jpeg"]:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, PNG, JPG and JPEG files are supported right now.",
        )

    regno_final = regno.strip() if regno else student_id

    # 1) Save uploaded file
    saved_path = os.path.join(UPLOAD_DIR, f"{student_id}_{file.filename}")
    contents = await file.read()
    with open(saved_path, "wb") as f:
        f.write(contents)

    # 2) Run transcript_ingestion.py via the venv interpreter
    parsed_transcript_path = "output/transcript_parsed_single.csv"

    if not os.path.exists(TRANSCRIPT_SCRIPT):
        raise HTTPException(
            status_code=500,
            detail=f"transcript_ingestion.py not found at {TRANSCRIPT_SCRIPT}",
        )

    cmd1 = [
        sys.executable,
        TRANSCRIPT_SCRIPT,
        "--file",
        saved_path,
        "--out-csv",
        parsed_transcript_path,
        "--student-id",
        student_id,
        "--regno",
        regno_final,
    ]
    res1 = subprocess.run(cmd1, capture_output=True, text=True)
    if res1.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"Transcript ingestion failed: {res1.stderr or res1.stdout}",
        )

    if not os.path.exists(parsed_transcript_path):
        raise HTTPException(
            status_code=500,
            detail="Parsed transcript CSV was not created.",
        )

    parsed_df = pd.read_csv(parsed_transcript_path)
    num_courses = len(parsed_df)

    # 3) Run skill_aggregation_from_parsed.py
    if not os.path.exists(AGGREGATION_SCRIPT):
        raise HTTPException(
            status_code=500,
            detail=f"skill_aggregation_from_parsed.py not found at {AGGREGATION_SCRIPT}",
        )

    cmd2 = [sys.executable, AGGREGATION_SCRIPT]
    res2 = subprocess.run(cmd2, capture_output=True, text=True)
    if res2.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"Skill aggregation failed: {res2.stderr or res2.stdout}",
        )

    skill_profile_path = "output/skill_profile_parsed_single.csv"
    if not os.path.exists(skill_profile_path):
        raise HTTPException(
            status_code=500,
            detail="Skill profile CSV was not created.",
        )

    skill_df = pd.read_csv(skill_profile_path)
    skill_df = skill_df[skill_df["StudentID"] == student_id]
    if skill_df.empty:
        raise HTTPException(
            status_code=500,
            detail="No skills produced for this transcript.",
        )

    if "ScoreNormalized" not in skill_df.columns:
        raise HTTPException(
            status_code=500,
            detail="Skill profile is missing 'ScoreNormalized' column.",
        )

    score_col = "ScoreNormalized"
    level_col = "SkillLevel" if "SkillLevel" in skill_df.columns else None

    # 4) Build skill list for response
    skill_df_sorted = skill_df.sort_values(score_col, ascending=False)
    skills_list: List[Dict[str, Any]] = []
    for _, row in skill_df_sorted.head(30).iterrows():
        skills_list.append(
            {
                "skill": row["Skill"],
                "score": float(row[score_col]),
                "level": str(row[level_col]) if level_col else "",
            }
        )

    # 5) Compute role matches using dynamic job templates
    try:
        templates_df = load_role_templates_dynamic()
        roles = compute_role_readiness_from_profile(
            skill_df, templates_df, weak_threshold=0.4, top_n=10
        )
        note = "Transcript processed successfully."
    except FileNotFoundError as e:
        roles = []
        note = str(e)

    return {
        "student_id": student_id,
        "regno": regno_final,
        "num_courses_parsed": int(num_courses),
        "num_skills": int(len(skill_df)),
        "skills": skills_list,
        "roles": roles,
        "note": note,
    }



@app.get("/students/{student_id}/roles")
def get_student_roles(student_id: str, top_n: int = 10) -> Dict[str, Any]:
    """
    Return top role matches for a student from role_readiness_dynamic.csv.
    """
    try:
        df = load_role_readiness()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    df = df[df["StudentID"] == student_id]
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No role readiness found for student {student_id}")

    df = df.sort_values("ReadinessScore", ascending=False).head(top_n)

    roles = [
        {
            "role_name": row["RoleName"],
            "readiness_score": float(row["ReadinessScore"]),
            "coverage": float(row.get("Coverage", 0.0)),
            "num_skills": int(row.get("NumSkills", 0)),
            "num_skills_present": int(row.get("NumSkillsPresent", 0)),
            "num_weak_or_missing": int(row.get("NumWeakOrMissing", 0)),
            "weak_or_missing_skills": str(row.get("WeakOrMissingSkills", "")),
        }
        for _, row in df.iterrows()
    ]

    return {
        "student_id": student_id,
        "count": len(roles),
        "roles": roles,
    }


@app.post("/students/{student_id}/prepare-quiz")
def prepare_quiz(student_id: str, max_questions: int = 5) -> Dict[str, Any]:
    """
    Return a small quiz for the student by sampling questions from
    output/quiz_questions_generated.csv.
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
            detail=f"No questions found for student {student_id}. "
                   f"Run quiz_generation_rag.py or relax filtering.",
        )

    if len(df) > max_questions:
        df = df.sample(n=max_questions, random_state=42)

    questions = []
    for _, row in df.iterrows():
        q = {
            "QuestionID": int(row["QuestionID"]),
            "QuestionText": str(row.get("QuestionText", "")),
            "Skill": str(row.get("Skill", "")),
            "Difficulty": str(row.get("Difficulty", "Unknown")),
            "RoleName": str(row.get("RoleName", "")),
            "OptionA": str(row.get("OptionA", "")),
            "OptionB": str(row.get("OptionB", "")),
            "OptionC": str(row.get("OptionC", "")),
            "OptionD": str(row.get("OptionD", "")),
        }
        questions.append(q)

    return {
        "student_id": student_id,
        "num_questions": len(questions),
        "questions": questions,
    }


@app.post("/students/{student_id}/submit-quiz")
def submit_quiz(student_id: str, submission: QuizSubmission) -> Dict[str, Any]:
    """
    Score quiz answers using quiz_questions_generated.csv and
    fuse results back into skill_profiles_with_quiz.csv, then
    recompute role readiness for that student.
    """
    try:
        df_q = load_quiz_questions()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if df_q.empty:
        raise HTTPException(status_code=500, detail="Question bank is empty.")

    if "QuestionID" not in df_q.columns:
        df_q = df_q.reset_index().rename(columns={"index": "QuestionID"})
    df_q = df_q.set_index("QuestionID")

    detailed_results = []
    correct_count = 0
    skill_stats: Dict[str, Dict[str, float]] = {}

    for r in submission.responses:
        qid = r.question_id
        if qid not in df_q.index:
            continue

        row = df_q.loc[qid]
        correct = str(row.get("CorrectOption", "")).strip().upper()
        selected = r.selected_option.strip().upper()
        skill = str(row.get("Skill", "Unknown"))

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
                "difficulty": str(row.get("Difficulty", "Unknown")),
                "role_name": str(row.get("RoleName", "")),
                "response_time_seconds": r.response_time_seconds,
            }
        )

        if skill not in skill_stats:
            skill_stats[skill] = {"num": 0, "correct": 0}
        skill_stats[skill]["num"] += 1
        if is_correct:
            skill_stats[skill]["correct"] += 1

    total_answered = sum(s["num"] for s in skill_stats.values())
    accuracy = correct_count / total_answered if total_answered > 0 else 0.0

    per_skill = []
    for skill, stats in skill_stats.items():
        per_skill.append(
            {
                "skill": skill,
                "num_questions": stats["num"],
                "num_correct": stats["correct"],
                "accuracy": stats["correct"] / stats["num"] if stats["num"] else 0.0,
            }
        )

    # Try to fuse quiz into skill profiles and recompute roles
    note_parts = []
    updated_roles_list: List[Dict[str, Any]] = []

    if per_skill:
        try:
            fused_profiles = fuse_quiz_into_skill_profiles(student_id, per_skill)
            new_roles = recompute_role_readiness_for_student(
                student_id, profiles_df=fused_profiles
            )
            if not new_roles.empty:
                new_roles = new_roles.sort_values("ReadinessScore", ascending=False).head(5)
                for _, r in new_roles.iterrows():
                    updated_roles_list.append(
                        {
                            "role_name": r["RoleName"],
                            "readiness_score": float(r["ReadinessScore"]),
                            "coverage": float(r.get("Coverage", 0.0)),
                            "num_skills": int(r.get("NumSkills", 0)),
                            "num_skills_present": int(r.get("NumSkillsPresent", 0)),
                            "num_weak_or_missing": int(r.get("NumWeakOrMissing", 0)),
                            "weak_or_missing_skills": str(r.get("WeakOrMissingSkills", "")),
                        }
                    )
            note_parts.append(
                "Skill profiles and role readiness updated using quiz results."
            )
        except Exception as e:
            # Do not crash the scoring if fusion fails
            note_parts.append(
                f"Quiz scored, but updating skill profiles / roles failed: {e}"
            )
    else:
        note_parts.append("No per-skill stats computed (no valid answers).")

    note = " ".join(note_parts)

    return {
        "student_id": student_id,
        "num_answered": total_answered,
        "num_correct": correct_count,
        "overall_accuracy": accuracy,
        "per_skill": per_skill,
        "detailed": detailed_results,
        "updated_roles": updated_roles_list,
        "note": note,
    }
