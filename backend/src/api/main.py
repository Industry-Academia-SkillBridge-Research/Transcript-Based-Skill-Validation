# backend/src/api/main.py

import os
import json
import re
from typing import Any, Dict, Optional, List, Tuple

import pandas as pd
import pdfplumber
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.transcript_ingestion import parse_transcript_text
from src.skill_aggregation_from_parsed import build_skill_profile_from_parsed
from src.rag_retriever import load_skill_context
from src.quiz_generation_gemini import generate_mcqs_from_context


app = FastAPI(
    title="Transcript-based Skill Validation API",
    version="0.3.2",
)

origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Request Models
# -----------------------------
class PrepareQuizRequest(BaseModel):
    selected_skills: List[str] = []
    num_questions_per_skill: int = 3
    difficulty: str = "mixed"  # "easy" | "medium" | "hard" | "mixed"


class QuizResponseItem(BaseModel):
    question_id: int
    selected_option: str
    response_time_seconds: Optional[float] = None


class SubmitQuizRequest(BaseModel):
    responses: List[QuizResponseItem]


# -----------------------------
# CSV Loaders
# -----------------------------
def load_role_readiness() -> pd.DataFrame:
    path = os.path.join("output", "role_readiness_dynamic.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Role readiness CSV not found. Run job_role_model_dynamic.py first.")
    return pd.read_csv(path)


def load_skill_explanations() -> pd.DataFrame:
    candidates = [
        "output/skill_explanations.csv",
        "output/skill_profiles_explainable.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError("No explainability file found. Generate explainable skill output first.")


def load_role_readiness_details() -> pd.DataFrame:
    candidates = [
        "output/role_readiness_details_dynamic.csv",
        "output/role_readiness_details_explainable.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError("Role readiness details file not found. Run job_role_model_dynamic.py first.")


def load_question_bank() -> pd.DataFrame:
    path = os.path.join("input", "question_bank.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("input/question_bank.csv not found. Create the question bank first.")

    df = pd.read_csv(path)

    required = [
        "QuestionID",
        "Skill",
        "Difficulty",
        "QuestionText",
        "OptionA",
        "OptionB",
        "OptionC",
        "OptionD",
        "CorrectOption",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"question_bank.csv missing columns: {missing}")

    return df


def save_quiz_answer_key(student_id: str, answer_key: dict):
    os.makedirs("output", exist_ok=True)
    path = os.path.join("output", f"quiz_answer_key_{student_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(answer_key, f, indent=2)


# -----------------------------
# Skill mapping helpers
# -----------------------------
def load_skill_aliases() -> List[Tuple[str, str]]:
    """
    Optional file: input/skill_aliases.csv with columns:
      pattern, bank_skill

    pattern can be a simple keyword or a regex.
    bank_skill must match a Skill value in question_bank.csv.
    """
    path = os.path.join("input", "skill_aliases.csv")
    if not os.path.exists(path):
        return []

    df = pd.read_csv(path)
    pairs: List[Tuple[str, str]] = []

    if "pattern" not in df.columns or "bank_skill" not in df.columns:
        return pairs

    for _, r in df.iterrows():
        pattern = str(r["pattern"]).strip()
        bank_skill = str(r["bank_skill"]).strip()
        if pattern and bank_skill:
            pairs.append((pattern, bank_skill))
    return pairs


def map_to_bank_skill(selected_skill: str, bank_skills: set, aliases: List[Tuple[str, str]]) -> Optional[str]:
    """
    Map a long transcript skill name to one of the bank skills.
    Priority:
      1) Exact match with bank skill
      2) Alias rules from input/skill_aliases.csv (regex search)
      3) Substring match: if "sql" appears in selected skill and bank has "SQL"
    """
    s = (selected_skill or "").strip()
    if not s:
        return None

    # 1) exact match
    if s in bank_skills:
        return s

    low = s.lower()

    # 2) alias match (regex / keyword)
    for pattern, bank_skill in aliases:
        try:
            if re.search(pattern.lower(), low):
                if bank_skill in bank_skills:
                    return bank_skill
        except re.error:
            # if invalid regex, fallback to substring
            if pattern.lower() in low and bank_skill in bank_skills:
                return bank_skill

    # 3) substring: bank skill appears inside selected skill
    for b in bank_skills:
        if str(b).lower() in low:
            return b

    return None


# -----------------------------
# Endpoints
# -----------------------------
@app.post("/students/{student_id}/upload-transcript")
async def upload_transcript(student_id: str, file: UploadFile = File(...)):
    try:
        os.makedirs("output", exist_ok=True)

        contents = await file.read()
        tmp_path = os.path.join("output", f"_tmp_{student_id}_{file.filename}")
        with open(tmp_path, "wb") as f:
            f.write(contents)

        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF supported for now.")

        text_parts = []
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
        text = "\n".join(text_parts)

        details, courses_df = parse_transcript_text(text)

        if courses_df.empty:
            raise HTTPException(
                status_code=422,
                detail="No course lines detected. Transcript format may not match parser regex.",
            )

        skills_df = build_skill_profile_from_parsed(
            student_id=student_id,
            parsed_courses_df=courses_df,
            mapping_path="input/course_skill_mapping.csv",
        )

        per_student_path = os.path.join("output", f"skill_profile_{student_id}.csv")
        try:
            skills_df.to_csv(per_student_path, index=False)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to write {per_student_path}: {e}")

        return {
            "student_id": student_id,
            "transcript_details": details,
            "num_courses_detected": int(len(courses_df)),
            "courses": courses_df.to_dict(orient="records"),
            "num_skills_mapped": int(len(skills_df)),
            "skills": skills_df.sort_values("ScoreNormalized", ascending=False).to_dict(orient="records"),
            "saved_skill_profile": os.path.basename(per_student_path),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing transcript: {e}")
    finally:
        pass


@app.get("/students/{student_id}/skills")
def get_student_skills(student_id: str):
    per_student_path = os.path.join("output", f"skill_profile_{student_id}.csv")
    path_primary = os.path.join("output", "skill_profiles_with_quiz.csv")
    path_fallback = os.path.join("output", "skill_profiles_explainable.csv")

    if os.path.exists(per_student_path):
        df = pd.read_csv(per_student_path)
        if "ScoreNormalized" in df.columns:
            df = df.sort_values("ScoreNormalized", ascending=False)
        return {
            "student_id": student_id,
            "source_file": os.path.basename(per_student_path),
            "skills": df.to_dict(orient="records"),
        }

    path = path_primary if os.path.exists(path_primary) else path_fallback
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Skill profile CSV not found in output/")

    df = pd.read_csv(path)

    col = "StudentID" if "StudentID" in df.columns else ("student_id" if "student_id" in df.columns else None)
    if col is None:
        raise HTTPException(status_code=500, detail=f"Student id column not found in {os.path.basename(path)}")

    student_df = df[df[col].astype(str) == str(student_id)].copy()
    if student_df.empty:
        raise HTTPException(
            status_code=404,
            detail="Student not found in skill profile. Upload transcript first or generate the global CSV.",
        )

    for score_col in ["FinalScore", "ScoreNormalized", "Score"]:
        if score_col in student_df.columns:
            student_df = student_df.sort_values(score_col, ascending=False)
            break

    return {
        "student_id": student_id,
        "source_file": os.path.basename(path),
        "skills": student_df.to_dict(orient="records"),
    }


@app.get("/students/{student_id}/roles")
def get_student_roles(student_id: str):
    path = os.path.join("output", "role_readiness_dynamic.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Role readiness CSV not found in output/")

    df = pd.read_csv(path)

    col = "StudentID" if "StudentID" in df.columns else ("student_id" if "student_id" in df.columns else None)
    if col is None:
        raise HTTPException(status_code=500, detail="Student id column not found in role readiness CSV")

    student_df = df[df[col].astype(str) == str(student_id)].copy()
    if student_df.empty:
        raise HTTPException(
            status_code=404,
            detail="Student not found in role readiness. Run role model pipeline for this student.",
        )

    for readiness_col in ["ReadinessScore", "RoleReadiness", "Score", "MatchScore"]:
        if readiness_col in student_df.columns:
            student_df = student_df.sort_values(readiness_col, ascending=False)
            break

    return {
        "student_id": student_id,
        "source_file": os.path.basename(path),
        "roles": student_df.to_dict(orient="records"),
    }


@app.get("/students/{student_id}/xai/skills")
def xai_skills(student_id: str, top_n: int = 15) -> Dict[str, Any]:
    try:
        df = load_skill_explanations()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if "StudentID" not in df.columns:
        raise HTTPException(status_code=500, detail="StudentID column missing in skill explanations file.")

    df_s = df[df["StudentID"].astype(str) == str(student_id)].copy()
    if df_s.empty:
        raise HTTPException(status_code=404, detail=f"No explainability found for student {student_id}")

    score_col = (
        "FinalScore"
        if "FinalScore" in df_s.columns
        else ("ScoreNormalized" if "ScoreNormalized" in df_s.columns else None)
    )
    if score_col:
        df_s = df_s.sort_values(score_col, ascending=False).head(int(top_n))
    else:
        df_s = df_s.head(int(top_n))

    evidence_cols = [c for c in ["Evidence", "EvidenceCourses", "CourseEvidence", "MatchedCourses"] if c in df_s.columns]

    out = []
    for _, r in df_s.iterrows():
        evidence_text = ""
        if evidence_cols:
            evidence_text = str(r[evidence_cols[0]])

        out.append(
            {
                "skill": str(r.get("Skill", "")),
                "score": float(r[score_col]) if score_col and pd.notna(r[score_col]) else None,
                "level": str(r["FinalSkillLevel"]) if "FinalSkillLevel" in df_s.columns else str(r.get("SkillLevel", "")),
                "evidence": evidence_text,
            }
        )

    return {"student_id": student_id, "count": len(out), "skills": out}


@app.get("/students/{student_id}/xai/roles")
def xai_roles(student_id: str, role: Optional[str] = None, top_n: int = 1) -> Dict[str, Any]:
    try:
        df = load_role_readiness_details()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if "StudentID" not in df.columns:
        raise HTTPException(status_code=500, detail="StudentID column missing in role readiness details file.")

    df_s = df[df["StudentID"].astype(str) == str(student_id)].copy()
    if df_s.empty:
        raise HTTPException(status_code=404, detail=f"No role details found for student {student_id}")

    if role is None:
        try:
            summary = load_role_readiness()
            if "StudentID" in summary.columns:
                s = summary[summary["StudentID"].astype(str) == str(student_id)].copy()
                if not s.empty:
                    readiness_col = "ReadinessScore" if "ReadinessScore" in s.columns else None
                    if readiness_col:
                        s = s.sort_values(readiness_col, ascending=False)
                    if "RoleName" in s.columns:
                        role = str(s.iloc[0]["RoleName"])
        except Exception:
            if "RoleName" in df_s.columns and not df_s.empty:
                role = str(df_s.iloc[0].get("RoleName", ""))

    if role and "RoleName" in df_s.columns:
        df_r = df_s[df_s["RoleName"].astype(str) == str(role)].copy()
    else:
        df_r = df_s.copy()

    if df_r.empty:
        raise HTTPException(status_code=404, detail=f"No detail rows for role '{role}' and student {student_id}")

    required_col = "RequiredImportance" if "RequiredImportance" in df_r.columns else None
    student_score_col = "StudentScore" if "StudentScore" in df_r.columns else None
    weak_col = "IsWeakOrMissing" if "IsWeakOrMissing" in df_r.columns else None

    rows = []
    for _, r in df_r.iterrows():
        rows.append(
            {
                "skill": str(r.get("Skill", "")),
                "required_importance": float(r[required_col]) if required_col and pd.notna(r.get(required_col)) else None,
                "student_score": float(r[student_score_col]) if student_score_col and pd.notna(r.get(student_score_col)) else None,
                "student_level": str(r.get("StudentLevel", "")),
                "attained_fraction": float(r["AttainedFraction"]) if "AttainedFraction" in df_r.columns and pd.notna(r.get("AttainedFraction")) else None,
                "is_weak_or_missing": bool(r[weak_col]) if weak_col and pd.notna(r.get(weak_col)) else None,
            }
        )

    weak_count = sum(1 for x in rows if x.get("is_weak_or_missing"))

    if top_n is not None and int(top_n) > 0:
        rows_sorted = sorted(
            rows,
            key=lambda x: (x["required_importance"] if x["required_importance"] is not None else -1.0),
            reverse=True,
        )
        rows = rows_sorted[: int(top_n)] if role is None else rows

    return {
        "student_id": student_id,
        "role_name": role,
        "num_required_skills": len(rows),
        "num_weak_or_missing": weak_count,
        "required_skills": rows,
    }


@app.post("/students/{student_id}/prepare-quiz")
def prepare_quiz(student_id: str, payload: PrepareQuizRequest):
    selected = [s.strip() for s in payload.selected_skills if s and s.strip()]
    if not selected:
        raise HTTPException(status_code=400, detail="Select at least 1 skill.")
    if len(selected) > 5:
        raise HTTPException(status_code=400, detail="You can select a maximum of 5 skills.")

    per_student_path = os.path.join("output", f"skill_profile_{student_id}.csv")
    if not os.path.exists(per_student_path):
        raise HTTPException(status_code=404, detail="Upload transcript first (skill profile not found).")

    skills_df = pd.read_csv(per_student_path)
    if "Skill" not in skills_df.columns:
        raise HTTPException(status_code=500, detail="Skill profile missing 'Skill' column.")

    available = set(skills_df["Skill"].astype(str))
    invalid = [s for s in selected if s not in available]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Invalid skills selected: {invalid}")

    difficulty = (payload.difficulty or "mixed").strip().lower()
    per_skill = int(payload.num_questions_per_skill or 3)

    # Load question bank for fallback
    try:
        qdf = load_question_bank()
        # Ensure numeric QuestionID
        qdf["QuestionID"] = pd.to_numeric(qdf["QuestionID"], errors="coerce")
        qdf = qdf.dropna(subset=["QuestionID"]).copy()
        qdf["QuestionID"] = qdf["QuestionID"].astype(int)
    except Exception:
        qdf = pd.DataFrame()

    bank_skills = set(qdf["Skill"].astype(str).unique()) if not qdf.empty else set()
    aliases = load_skill_aliases() if "load_skill_aliases" in globals() else []

    questions = []
    answer_key = {}
    missing_skills = []

    next_qid = 100000  # ids for Gemini-generated questions

    for selected_skill in selected:
        # Map long skill -> bank skill key (SQL/Java/etc.) to decide which KB file to use
        mapped_skill = None
        if bank_skills:
            mapped_skill = map_to_bank_skill(selected_skill, bank_skills, aliases)

        # If bank is empty or mapping fails, use a simple default key:
        # take first word (e.g., "Java Programming..." -> "Java")
        if not mapped_skill:
            mapped_skill = selected_skill.split()[0].strip() if selected_skill.split() else selected_skill

        # --- 1) Try Gemini + RAG ---
        gemini_ok = False
        try:
            context = load_skill_context(mapped_skill, kb_dir="knowledge_base", max_chars=5500)
            if context and context.strip():
                gen = generate_mcqs_from_context(
                    skill_key=mapped_skill,
                    context=context,
                    n=per_skill,
                    difficulty=difficulty
                )

                gen_questions = gen.get("questions", [])
                for q in gen_questions:
                    opts = q.get("options", {}) or {}
                    ans = str(q.get("answer", "")).strip().upper()

                    # Validate minimal structure
                    if ans not in ["A", "B", "C", "D"]:
                        continue
                    if not all(k in opts for k in ["A", "B", "C", "D"]):
                        continue

                    qid = next_qid
                    next_qid += 1

                    answer_key[str(qid)] = ans

                    questions.append(
                        {
                            "QuestionID": qid,
                            "SelectedSkill": selected_skill,
                            "Skill": mapped_skill,
                            "Difficulty": difficulty,
                            "QuestionText": str(q.get("question", "")).strip(),
                            "OptionA": str(opts.get("A", "")).strip(),
                            "OptionB": str(opts.get("B", "")).strip(),
                            "OptionC": str(opts.get("C", "")).strip(),
                            "OptionD": str(opts.get("D", "")).strip(),
                            "Source": "gemini",
                        }
                    )

                if len([x for x in questions if x["SelectedSkill"] == selected_skill]) > 0:
                    gemini_ok = True
        except Exception:
            gemini_ok = False

        # --- 2) Fallback to Question Bank if Gemini failed ---
        if not gemini_ok:
            if qdf.empty:
                missing_skills.append(selected_skill)
                continue

            # Use mapped_skill to pull from bank (SQL/Java/etc.)
            subset = qdf[qdf["Skill"].astype(str) == str(mapped_skill)].copy()
            if difficulty != "mixed":
                subset = subset[subset["Difficulty"].astype(str).str.lower() == difficulty]

            if subset.empty:
                missing_skills.append(selected_skill)
                continue

            subset = subset.sample(n=min(per_skill, len(subset)), random_state=42)

            for _, row in subset.iterrows():
                qid = int(row["QuestionID"])
                answer_key[str(qid)] = str(row["CorrectOption"]).strip().upper()

                questions.append(
                    {
                        "QuestionID": qid,
                        "SelectedSkill": selected_skill,
                        "Skill": str(row["Skill"]),
                        "Difficulty": str(row["Difficulty"]),
                        "QuestionText": str(row["QuestionText"]),
                        "OptionA": str(row["OptionA"]),
                        "OptionB": str(row["OptionB"]),
                        "OptionC": str(row["OptionC"]),
                        "OptionD": str(row["OptionD"]),
                        "Source": "bank",
                    }
                )

    if missing_skills:
        raise HTTPException(
            status_code=404,
            detail=f"No questions could be generated for: {missing_skills}. Add knowledge_base files or expand question_bank.csv.",
        )

    save_quiz_answer_key(student_id, answer_key)

    return {
        "student_id": student_id,
        "selected_skills": selected,
        "num_questions": len(questions),
        "questions": questions,
    }



@app.post("/students/{student_id}/submit-quiz")
def submit_quiz(student_id: str, payload: SubmitQuizRequest):
    key_path = os.path.join("output", f"quiz_answer_key_{student_id}.json")
    if not os.path.exists(key_path):
        raise HTTPException(status_code=404, detail="Answer key not found. Generate a quiz first.")

    with open(key_path, "r", encoding="utf-8") as f:
        answer_key = json.load(f)

    total = len(payload.responses)
    correct = 0
    per_question = []

    for r in payload.responses:
        qid = str(r.question_id)
        gold = str(answer_key.get(qid, "")).upper()
        picked = str(r.selected_option).upper()

        is_correct = picked == gold and gold in ["A", "B", "C", "D"]
        if is_correct:
            correct += 1

        per_question.append(
            {
                "question_id": r.question_id,
                "selected_option": picked,
                "correct_option": gold,
                "is_correct": is_correct,
            }
        )

    accuracy = (correct / total) if total else 0.0

    return {
        "student_id": student_id,
        "total_answered": total,
        "correct": correct,
        "accuracy": accuracy,
        "per_question": per_question,
    }
