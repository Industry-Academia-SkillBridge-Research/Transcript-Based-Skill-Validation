import os
from pathlib import Path
import sys
import subprocess
from typing import List, Dict
import shutil
from tempfile import NamedTemporaryFile

import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from src import transcript_ingestion

BASE_DIR = Path(__file__).resolve().parents[2]  # project root (was parents[1] causing src/src path)

app = FastAPI(
    title="Transcript-based Skill Validation API",
    version="0.1.0",
    description="Backend API over the skill validation and job alignment pipeline.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # later you can restrict to your real frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

    # ---- make columns robust to schema changes ----
    # BaselineScore: if missing, use ScoreNormalized
    if "BaselineScore" not in stu.columns:
        if "ScoreNormalized" in stu.columns:
            stu["BaselineScore"] = stu["ScoreNormalized"]
        else:
            stu["BaselineScore"] = 0.0

    # FinalScore: if missing, fall back to ScoreNormalized
    if "FinalScore" not in stu.columns:
        if "ScoreNormalized" in stu.columns:
            stu["FinalScore"] = stu["ScoreNormalized"]
        else:
            stu["FinalScore"] = 0.0

    # FinalSkillLevel: if missing, fall back to SkillLevel
    if "FinalSkillLevel" not in stu.columns:
        if "SkillLevel" in stu.columns:
            stu["FinalSkillLevel"] = stu["SkillLevel"]
        else:
            stu["FinalSkillLevel"] = "Unknown"

    # -----------------------------------------------

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

@app.post("/students/{student_id}/upload-transcript")
async def upload_transcript(
    student_id: str,
    regno: str = Form(...),
    file: UploadFile = File(...)
):
    """
    1) Save the uploaded transcript (pdf/image) to a temp file
    2) Extract text + parse courses
    3) Aggregate skills for this student (using your existing script)
    4) Merge into global skill_profiles_with_quiz.csv
    5) Recompute role readiness using job_role_model_dynamic.py
    6) Return parsed courses + top skills + top roles
    """

    # 1) Save uploaded file to a temporary location
    os.makedirs("uploads", exist_ok=True)
    suffix = os.path.splitext(file.filename)[1] or ".pdf"
    with NamedTemporaryFile(delete=False, dir="uploads", suffix=suffix) as tmp:
        tmp_path = tmp.name
        shutil.copyfileobj(file.file, tmp)

    # 2) Extract text + parse courses using your transcript_ingestion module
    try:
        text = transcript_ingestion.extract_text_from_file(tmp_path)
        parsed_df = transcript_ingestion.parse_transcript_text(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcript parsing failed: {e}")

    if parsed_df.empty:
        raise HTTPException(status_code=400, detail="Could not parse any course rows from transcript")

    # Fill student metadata
    parsed_df["StudentID"] = student_id
    parsed_df["RegNo"] = regno

    # Save in the location expected by skill_aggregation_from_parsed.py
    os.makedirs("output", exist_ok=True)
    parsed_csv_path = os.path.join("output", "transcript_parsed_single.csv")
    parsed_df.to_csv(parsed_csv_path, index=False)

    # 3) Run your existing skill aggregation for parsed transcript
    skill_agg_script = os.path.join("src", "skill_aggregation_from_parsed.py")
    result = subprocess.run(
        [sys.executable, skill_agg_script],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"Skill aggregation failed:\n{result.stderr}"
        )

    skill_single_path = os.path.join("output", "skill_profile_parsed_single.csv")
    if not os.path.exists(skill_single_path):
        raise HTTPException(status_code=500, detail="Skill profile file not found after aggregation.")

    skill_single_df = pd.read_csv(skill_single_path)

    # 4) Merge this student's skills into global skill_profiles_with_quiz.csv
    fused_path = os.path.join("output", "skill_profiles_with_quiz.csv")
    if not os.path.exists(fused_path):
        raise HTTPException(
            status_code=500,
            detail="Global skill_profiles_with_quiz.csv not found. Run your offline pipeline first."
        )

    fused_df = pd.read_csv(fused_path)

    # Build rows with same columns as fused_df
    new_rows = []
    for _, row in skill_single_df.iterrows():
        rec = {}
        for col in fused_df.columns:
            if col in skill_single_df.columns:
                rec[col] = row[col]
            elif col == "FinalScore":
                rec[col] = float(row.get("ScoreNormalized", 0.0))
            elif col == "FinalSkillLevel":
                rec[col] = row.get("SkillLevel", "Beginner")
            elif col in [
                "QuizQuestionCount",
                "AvgQuestionScore",
                "AvgEffectiveScore",
                "QuizProficiency",
                "QuizProficiencyFilled",
            ]:
                rec[col] = 0.0
            else:
                # Anything else gets a safe default
                rec[col] = None
        new_rows.append(rec)

    new_rows_df = pd.DataFrame(new_rows)

    # Drop old rows for this student (if any), then append new ones
    fused_df = fused_df[fused_df["StudentID"] != student_id]
    fused_df = pd.concat([fused_df, new_rows_df], ignore_index=True)
    fused_df.to_csv(fused_path, index=False)

    # 5) Recompute role readiness using your dynamic role model script
    role_model_script = os.path.join("src", "job_role_model_dynamic.py")
    result2 = subprocess.run(
        [sys.executable, role_model_script],
        capture_output=True,
        text=True,
    )
    if result2.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"Role readiness computation failed:\n{result2.stderr}"
        )

    # 6) Prepare response: parsed courses, top skills, top roles
    #   a) courses
    courses_preview = parsed_df[["CourseCode", "CourseTitle", "Grade"]].fillna("").to_dict(
        orient="records"
    )

    #   b) skills for this student from fused file
    fused_df = pd.read_csv(fused_path)
    student_skills = fused_df[fused_df["StudentID"] == student_id].copy()
    student_skills = student_skills.sort_values(
        by=student_skills.columns[-1], ascending=False
    )  # last col usually FinalScore
    skills_preview = student_skills[["Skill", "FinalScore", "FinalSkillLevel"]].head(15).to_dict(
        orient="records"
    )

    #   c) roles from updated role_readiness_dynamic.csv
    roles_path = os.path.join("output", "role_readiness_dynamic.csv")
    roles_preview = []
    if os.path.exists(roles_path):
        roles_df = pd.read_csv(roles_path)
        roles_df = roles_df[roles_df["StudentID"] == student_id].copy()
        roles_df = roles_df.sort_values("ReadinessScore", ascending=False).head(10)
        roles_preview = roles_df[
            ["RoleName", "ReadinessScore", "Coverage", "WeakOrMissingSkills"]
        ].to_dict(orient="records")

    return {
        "message": "Transcript processed and profiles updated",
        "student_id": student_id,
        "parsed_courses_count": int(len(parsed_df)),
        "skills_count": int(len(student_skills)),
        "courses": courses_preview,
        "top_skills": skills_preview,
        "top_roles": roles_preview,
    }
