from pathlib import Path
import sys
import subprocess
from typing import List, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parents[1]

app = FastAPI(
    title="Transcript-based Skill Validation API",
    version="0.1.0",
    description="Backend API over the skill validation and job alignment pipeline.",
)


# ---------- Pydantic models ----------

class QuestionItem(BaseModel):
    question_id: str
    role_name: str
    skill: str
    difficulty: str
    target_difficulty: str | None = None
    question_text: str
    options: Dict[str, str]


class QuizStartResponse(BaseModel):
    student_id: str
    questions: List[QuestionItem]


class QuizAnswer(BaseModel):
    question_id: str
    selected_option: str
    response_time_seconds: float | None = None


class QuizSubmission(BaseModel):
    student_id: str
    answers: List[QuizAnswer]


class SkillSummary(BaseModel):
    skill: str
    baseline_score: float | None = None
    final_score: float | None = None
    level: str | None = None


class RoleSummary(BaseModel):
    role_name: str
    readiness_score: float
    coverage: float
    num_skills: int
    num_weak_or_missing: int
    weak_or_missing_skills: str | None = None


class QuizSubmitResponse(BaseModel):
    student_id: str
    num_answers: int
    top_skills: List[SkillSummary]
    top_roles: List[RoleSummary]


# ---------- Helpers ----------

def run_pipeline_for_student(student_id: str, phase: str) -> None:
    """
    Call the orchestration script with the same Python interpreter.
    """
    script_path = BASE_DIR / "src" / "run_full_pipeline_for_student.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--student-id",
        student_id,
        "--phase",
        phase,
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed for phase={phase}: {e}",
        )


def load_quiz_questions_for_student(student_id: str) -> pd.DataFrame:
    questions_path = BASE_DIR / "output" / "quiz_questions_generated.csv"
    if not questions_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Quiz questions not found. Run pre-quiz pipeline first.",
        )
    df = pd.read_csv(questions_path)
    if "StudentID" not in df.columns:
        raise HTTPException(
            status_code=500,
            detail="Questions file missing 'StudentID' column.",
        )
    stu_df = df[df["StudentID"] == student_id].copy()
    if stu_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No quiz questions found for student {student_id}.",
        )
    # ensure required columns exist
    required_cols = {
        "QuestionID",
        "RoleName",
        "Skill",
        "Difficulty",
        "QuestionText",
        "OptionA",
        "OptionB",
        "OptionC",
        "OptionD",
    }
    missing = required_cols - set(stu_df.columns)
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Questions file missing columns: {missing}",
        )
    return stu_df


def append_quiz_responses(submission: QuizSubmission) -> None:
    """
    Append responses to the CSV that quiz_scoring.py expects.
    Columns: StudentID, QuestionID, SelectedOption, ResponseTimeSeconds
    """
    responses_path = BASE_DIR / "output" / "quiz_responses_sample.csv"
    rows = []
    for ans in submission.answers:
        rows.append(
            {
                "StudentID": submission.student_id,
                "QuestionID": ans.question_id,
                "SelectedOption": ans.selected_option.upper().strip(),
                "ResponseTimeSeconds": ans.response_time_seconds
                if ans.response_time_seconds is not None
                else "",
            }
        )
    new_df = pd.DataFrame(rows)

    if responses_path.exists():
        existing = pd.read_csv(responses_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.to_csv(responses_path, index=False)
    else:
        new_df.to_csv(responses_path, index=False)


def get_top_skills_for_student(student_id: str, top_k: int = 10) -> List[SkillSummary]:
    fused_path = BASE_DIR / "output" / "skill_profiles_with_quiz.csv"
    if not fused_path.exists():
        return []

    df = pd.read_csv(fused_path)
    if "StudentID" not in df.columns:
        return []

    stu = df[df["StudentID"] == student_id].copy()
    if stu.empty:
        return []

    # these columns come from skill_profile_fusion.py
    # BaselineScore = ScoreNormalized
    # FinalScore = fused score
    for col in ["BaselineScore", "FinalScore", "FinalSkillLevel"]:
        if col not in stu.columns:
            return []

    stu = stu.sort_values("FinalScore", ascending=False).head(top_k)

    results: List[SkillSummary] = []
    for _, row in stu.iterrows():
        results.append(
            SkillSummary(
                skill=str(row["Skill"]),
                baseline_score=float(row["BaselineScore"]),
                final_score=float(row["FinalScore"]),
                level=str(row["FinalSkillLevel"]),
            )
        )
    return results


def get_top_roles_for_student(student_id: str, top_k: int = 5) -> List[RoleSummary]:
    roles_path = BASE_DIR / "output" / "role_readiness_dynamic.csv"
    if not roles_path.exists():
        return []

    df = pd.read_csv(roles_path)
    if "StudentID" not in df.columns:
        return []

    stu = df[df["StudentID"] == student_id].copy()
    if stu.empty:
        return []

    required_cols = [
        "RoleName",
        "ReadinessScore",
        "Coverage",
        "NumSkills",
        "NumWeakOrMissing",
    ]
    for col in required_cols:
        if col not in stu.columns:
            return []

    stu = stu.sort_values("ReadinessScore", ascending=False).head(top_k)

    results: List[RoleSummary] = []
    for _, row in stu.iterrows():
        results.append(
            RoleSummary(
                role_name=str(row["RoleName"]),
                readiness_score=float(row["ReadinessScore"]),
                coverage=float(row["Coverage"]),
                num_skills=int(row["NumSkills"]),
                num_weak_or_missing=int(row["NumWeakOrMissing"]),
                weak_or_missing_skills=str(
                    row["WeakOrMissingSkills"]
                ) if "WeakOrMissingSkills" in stu.columns else None,
            )
        )
    return results


# ---------- Endpoints ----------


@app.post("/students/{student_id}/prepare-quiz", response_model=QuizStartResponse)
def prepare_quiz(student_id: str):
    """
    Run the pre_quiz pipeline for a given student and return
    the generated quiz questions for that student.
    """
    # 1) Run full pre-quiz pipeline (transcript → skills → jobs → quiz)
    run_pipeline_for_student(student_id, phase="pre_quiz")

    # 2) Load questions for this student
    stu_df = load_quiz_questions_for_student(student_id)

    questions: List[QuestionItem] = []
    for _, row in stu_df.iterrows():
        questions.append(
            QuestionItem(
                question_id=str(row["QuestionID"]),
                role_name=str(row["RoleName"]),
                skill=str(row["Skill"]),
                difficulty=str(row.get("Difficulty", "Unknown")),
                target_difficulty=str(row.get("TargetDifficulty", "")),
                question_text=str(row["QuestionText"]),
                options={
                    "A": str(row["OptionA"]),
                    "B": str(row["OptionB"]),
                    "C": str(row["OptionC"]),
                    "D": str(row["OptionD"]),
                },
            )
        )

    if not questions:
        raise HTTPException(
            status_code=404,
            detail=f"No questions available for student {student_id}.",
        )

    return QuizStartResponse(student_id=student_id, questions=questions)


@app.post("/students/{student_id}/submit-quiz", response_model=QuizSubmitResponse)
def submit_quiz(student_id: str, submission: QuizSubmission):
    """
    Accept quiz answers, score them, update skill profile, and recompute job-role alignment.
    """
    if submission.student_id != student_id:
        raise HTTPException(
            status_code=400,
            detail="StudentID in path and body do not match.",
        )

    if not submission.answers:
        raise HTTPException(
            status_code=400,
            detail="No answers provided.",
        )

    # 1) Append responses to CSV
    append_quiz_responses(submission)

    # 2) Run post-quiz pipeline (quiz_scoring + fusion + new readiness)
    run_pipeline_for_student(student_id, phase="post_quiz")

    # 3) Return updated skill and role summaries
    top_skills = get_top_skills_for_student(student_id)
    top_roles = get_top_roles_for_student(student_id)

    return QuizSubmitResponse(
        student_id=student_id,
        num_answers=len(submission.answers),
        top_skills=top_skills,
        top_roles=top_roles,
    )


@app.get("/students/{student_id}/skills", response_model=List[SkillSummary])
def get_skills(student_id: str):
    """
    Get fused skill profile (transcript + quiz) for a student.
    """
    skills = get_top_skills_for_student(student_id, top_k=50)
    if not skills:
        raise HTTPException(
            status_code=404,
            detail=f"No fused skill profile found for student {student_id}.",
        )
    return skills


@app.get("/students/{student_id}/roles", response_model=List[RoleSummary])
def get_roles(student_id: str):
    """
    Get top job roles (from real job postings) for a student.
    """
    roles = get_top_roles_for_student(student_id, top_k=20)
    if not roles:
        raise HTTPException(
            status_code=404,
            detail=f"No role readiness data found for student {student_id}.",
        )
    return roles
