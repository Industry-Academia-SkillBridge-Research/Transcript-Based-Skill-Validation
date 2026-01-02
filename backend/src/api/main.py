# backend/src/api/main.py

import os
import json
import re
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, List, Tuple

import pandas as pd
import pdfplumber
from PIL import Image
import pytesseract
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
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
    include_related: bool = True
    max_total_skills: int = 5


class QuizResponseItem(BaseModel):
    question_id: int
    selected_option: str
    response_time_seconds: Optional[float] = None


class SubmitQuizRequest(BaseModel):
    responses: List[QuizResponseItem]
    quiz_id: Optional[str] = None  # Optional quiz_id for quiz-based submission
    session_token: Optional[str] = None  # Quiz session token for time validation
    violations: Optional[List[Dict[str, Any]]] = []  # Security violations logged on frontend


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


def find_related_skills(selected: List[str], qdf: pd.DataFrame, max_total: int = 5) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Expand selected skills with related skills found in the question bank (qdf).
    Returns (expanded_list, map_of_origin_to_added_related_skills).
    Simple heuristic: token overlap and substring match (case-insensitive).
    """
    if qdf is None or qdf.empty:
        return selected[:max_total], {s: [] for s in selected}

    # Unique bank skills
    bank_skills = sorted(set(qdf["Skill"].astype(str)))
    selected_clean = [s.strip() for s in selected if s and s.strip()]

    # Tokenize helper
    def tokens(s: str):
        return {t for t in re.split(r"\W+", s.lower()) if len(t) > 2}

    expanded = []
    added_set = set()
    related_map: Dict[str, List[str]] = {}

    # Start by adding the original selected skills (if they exist in bank, prefer bank naming)
    for s in selected_clean:
        # Try to find exact bank skill name first
        bank_match = None
        for b in bank_skills:
            if s == b or s.lower() == b.lower():
                bank_match = b
                break
        entry = bank_match if bank_match else s
        if entry not in added_set:
            expanded.append(entry)
            added_set.add(entry)
        related_map[s] = []

        if len(added_set) >= max_total:
            break

    if len(added_set) < max_total:
        # Find candidates for each selected
        candidates_scores: Dict[str, float] = {}
        for s in selected_clean:
            s_toks = tokens(s)
            for b in bank_skills:
                if b in added_set:
                    continue
                b_toks = tokens(b)
                score = len(s_toks & b_toks)
                # substring boost
                if s.lower() in b.lower() or b.lower() in s.lower():
                    score += 1
                if score > 0:
                    candidates_scores.setdefault(b, 0.0)
                    candidates_scores[b] = max(candidates_scores[b], score)

        # Sort candidates by score desc, then by name
        sorted_cands = sorted(candidates_scores.items(), key=lambda x: (-x[1], x[0]))

        # Greedily add highest scoring candidates, but also assign to one of the origins (closest token overlap)
        for cand, _ in sorted_cands:
            if len(added_set) >= max_total:
                break
            # Find best origin to attach this candidate to
            best_origin = None
            best_score = -1
            cand_toks = tokens(cand)
            for s in selected_clean:
                score = len(tokens(s) & cand_toks)
                if s.lower() in cand.lower() or cand.lower() in s.lower():
                    score += 1
                if score > best_score:
                    best_score = score
                    best_origin = s
            if cand not in added_set and best_score > 0:
                expanded.append(cand)
                added_set.add(cand)
                if best_origin:
                    related_map.setdefault(best_origin, []).append(cand)

    # Ensure we return at most max_total and preserve order (originals first)
    expanded = expanded[:max_total]
    return expanded, related_map


# -----------------------------
# Endpoints
# -----------------------------
@app.post("/upload-transcript")
async def upload_transcript_auto(file: UploadFile = File(...), student_id: Optional[str] = Form(None)):
    """
    Upload transcript without requiring student ID in the path.
    Student ID will be extracted from the transcript if not provided.
    """
    try:
        os.makedirs("output", exist_ok=True)

        contents = await file.read()
        
        # Determine file type
        filename_lower = file.filename.lower()
        is_pdf = filename_lower.endswith(".pdf")
        is_image = any(filename_lower.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"])
        
        if not (is_pdf or is_image):
            raise HTTPException(
                status_code=400,
                detail="Only PDF and image files (JPG, PNG, GIF, WEBP, BMP) are supported."
            )

        # Extract text from file
        import tempfile
        import io
        
        text_parts = []
        
        if is_pdf:
            # Handle PDF files
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(contents)
                tmp_path = tmp_file.name
            
            with pdfplumber.open(tmp_path) as pdf:
                for page in pdf.pages:
                    text_parts.append(page.extract_text() or "")
            
            # Clean up temp PDF file
            try:
                os.unlink(tmp_path)
            except:
                pass
        else:
            # Handle image files using OCR
            try:
                image = Image.open(io.BytesIO(contents))
                # Convert to RGB if necessary (for formats like PNG with transparency)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Use pytesseract to extract text
                text = pytesseract.image_to_string(image, lang='eng')
                text_parts.append(text)
            except Exception as e:
                raise HTTPException(
                    status_code=422,
                    detail=f"Failed to process image with OCR: {str(e)}"
                )
        
        text = "\n".join(text_parts)

        details, courses_df = parse_transcript_text(text)

        if courses_df.empty:
            raise HTTPException(
                status_code=422,
                detail="No course lines detected. Transcript format may not match parser regex.",
            )

        # Use provided student_id, or extract from transcript, or generate one
        extracted_student_id = details.get("student_id") or details.get("registration_number") or ""
        # Clean up student_id from form (could be None, empty string, or whitespace)
        provided_student_id = (student_id or "").strip() if student_id else ""
        final_student_id = (provided_student_id or extracted_student_id).strip()
        
        if not final_student_id:
            # Generate a temporary ID based on filename and timestamp
            import hashlib
            import time
            filename_hash = hashlib.md5(file.filename.encode()).hexdigest()[:8]
            timestamp = str(int(time.time()))[-6:]
            final_student_id = f"TEMP_{filename_hash}_{timestamp}"
            details["student_id"] = final_student_id
            details["is_temporary_id"] = True

        # Update details with final student ID
        details["student_id"] = final_student_id
        if not details.get("registration_number"):
            details["registration_number"] = final_student_id

        # Create uploads directory and save original file
        uploads_dir = os.path.join("output", "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        saved_file_path = os.path.join(uploads_dir, f"{final_student_id}.pdf" if is_pdf else f"{final_student_id}.{filename_lower.split('.')[-1]}")
        with open(saved_file_path, "wb") as f:
            f.write(contents)
        
        # Also save temp file for processing (cleaned up later)
        final_tmp_path = os.path.join("output", f"_tmp_{final_student_id}_{file.filename}")
        with open(final_tmp_path, "wb") as f:
            f.write(contents)

        # Infer year from course code (IT1xxx = Year 1, IT2xxx = Year 2, etc.)
        def infer_year_from_code(code: str) -> Optional[int]:
            if not code or len(code) < 4:
                return None
            try:
                # Extract first digit after "IT"
                if code.upper().startswith("IT") and code[2].isdigit():
                    year = int(code[2])
                    if 1 <= year <= 4:
                        return year
            except (ValueError, IndexError):
                pass
            return None

        if "Year" not in courses_df.columns:
            courses_df["Year"] = courses_df["CourseCode"].apply(infer_year_from_code)
        
        # Add StudentID column for saving parsed transcript
        courses_df["StudentID"] = final_student_id
        
        # Save parsed transcript to transcript_parsed_single.csv
        parsed_transcript_path = os.path.join("output", "transcript_parsed_single.csv")
        # Select columns in order: StudentID, CourseCode, CourseTitle, Grade, Year
        columns_to_save = ["StudentID", "CourseCode", "CourseTitle", "Grade"]
        if "Year" in courses_df.columns:
            columns_to_save.append("Year")
        
        # Save to CSV (append if file exists, otherwise create new)
        if os.path.exists(parsed_transcript_path):
            # Read existing file and append
            existing_df = pd.read_csv(parsed_transcript_path)
            # Remove old entries for this student (if any)
            existing_df = existing_df[existing_df["StudentID"] != final_student_id]
            # Combine with new data
            combined_df = pd.concat([existing_df, courses_df[columns_to_save]], ignore_index=True)
            combined_df.to_csv(parsed_transcript_path, index=False)
        else:
            # Create new file
            courses_df[columns_to_save].to_csv(parsed_transcript_path, index=False)

        # Calculate statistics
        GRADE_POINTS = {
            "A+": 4.0, "A": 4.0, "A-": 3.7,
            "B+": 3.3, "B": 3.0, "B-": 2.7,
            "C+": 2.3, "C": 2.0, "C-": 1.7,
            "D+": 1.3, "D": 1.0, "D-": 0.7,
            "E": 0.0, "F": 0.0,
        }
        
        courses_df["GradePoint"] = courses_df["Grade"].str.upper().map(GRADE_POINTS).fillna(0.0)
        valid_grades = courses_df[courses_df["GradePoint"] > 0]
        
        # Calculate grade distribution
        grade_dist = courses_df["Grade"].astype(str).str.upper().str.replace(r"[+-]", "", regex=True)
        grade_distribution = grade_dist.value_counts().to_dict()
        # Convert numpy types to native Python types
        grade_distribution = {str(k): int(v) for k, v in grade_distribution.items()}
        
        stats = {
            "total_courses": int(len(courses_df)),
            "average_gpa": float(valid_grades["GradePoint"].mean()) if len(valid_grades) > 0 else None,
            "grade_distribution": grade_distribution,
        }

        skills_df = build_skill_profile_from_parsed(
            student_id=final_student_id,
            parsed_courses_df=courses_df,
            mapping_path="input/course_skill_mapping.csv",
        )

        per_student_path = os.path.join("output", f"skill_profile_{final_student_id}.csv")
        try:
            skills_df.to_csv(per_student_path, index=False)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to write {per_student_path}: {e}")
        
        # Also save to skill_profile_parsed_single.csv (single student format)
        parsed_single_path = os.path.join("output", "skill_profile_parsed_single.csv")
        skills_df.to_csv(parsed_single_path, index=False)

        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass

        return {
            "student_id": final_student_id,
            "transcript_details": details,
            "num_courses_detected": int(len(courses_df)),
            "courses": courses_df.to_dict(orient="records"),
            "statistics": stats,
            "num_skills_mapped": int(len(skills_df)),
            "skills": skills_df.sort_values("ScoreNormalized", ascending=False).to_dict(orient="records"),
            "saved_skill_profile": os.path.basename(per_student_path),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing transcript: {e}")


@app.post("/students/{student_id}/upload-transcript")
async def upload_transcript(student_id: str, file: UploadFile = File(...)):
    try:
        os.makedirs("output", exist_ok=True)

        contents = await file.read()
        tmp_path = os.path.join("output", f"_tmp_{student_id}_{file.filename}")
        with open(tmp_path, "wb") as f:
            f.write(contents)

        filename_lower = file.filename.lower()
        is_pdf = filename_lower.endswith(".pdf")
        is_image = any(filename_lower.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"])
        
        if not (is_pdf or is_image):
            raise HTTPException(
                status_code=400,
                detail="Only PDF and image files (JPG, PNG, GIF, WEBP, BMP) are supported."
            )

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
        
        # Save original file to uploads directory
        uploads_dir = os.path.join("output", "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        saved_file_path = os.path.join(uploads_dir, f"{student_id}.pdf")
        with open(saved_file_path, "wb") as f:
            f.write(contents)

        # Infer year from course code (IT1xxx = Year 1, IT2xxx = Year 2, etc.)
        def infer_year_from_code(code: str) -> Optional[int]:
            if not code or len(code) < 4:
                return None
            try:
                # Extract first digit after "IT"
                if code.upper().startswith("IT") and code[2].isdigit():
                    year = int(code[2])
                    if 1 <= year <= 4:
                        return year
            except (ValueError, IndexError):
                pass
            return None

        if "Year" not in courses_df.columns:
            courses_df["Year"] = courses_df["CourseCode"].apply(infer_year_from_code)
        
        # Save original file to uploads directory
        uploads_dir = os.path.join("output", "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        saved_file_path = os.path.join(uploads_dir, f"{student_id}.pdf")
        with open(saved_file_path, "wb") as f:
            f.write(contents)
        
        # Add StudentID column for saving parsed transcript
        courses_df["StudentID"] = student_id
        
        # Save parsed transcript to transcript_parsed_single.csv
        parsed_transcript_path = os.path.join("output", "transcript_parsed_single.csv")
        # Select columns in order: StudentID, CourseCode, CourseTitle, Grade, Year
        columns_to_save = ["StudentID", "CourseCode", "CourseTitle", "Grade"]
        if "Year" in courses_df.columns:
            columns_to_save.append("Year")
        
        # Save to CSV (append if file exists, otherwise create new)
        if os.path.exists(parsed_transcript_path):
            # Read existing file and append
            existing_df = pd.read_csv(parsed_transcript_path)
            # Remove old entries for this student (if any)
            existing_df = existing_df[existing_df["StudentID"] != student_id]
            # Combine with new data
            combined_df = pd.concat([existing_df, courses_df[columns_to_save]], ignore_index=True)
            combined_df.to_csv(parsed_transcript_path, index=False)
        else:
            # Create new file
            courses_df[columns_to_save].to_csv(parsed_transcript_path, index=False)

        # Calculate statistics
        GRADE_POINTS = {
            "A+": 4.0, "A": 4.0, "A-": 3.7,
            "B+": 3.3, "B": 3.0, "B-": 2.7,
            "C+": 2.3, "C": 2.0, "C-": 1.7,
            "D+": 1.3, "D": 1.0, "D-": 0.7,
            "E": 0.0, "F": 0.0,
        }
        
        courses_df["GradePoint"] = courses_df["Grade"].str.upper().map(GRADE_POINTS).fillna(0.0)
        valid_grades = courses_df[courses_df["GradePoint"] > 0]
        
        # Calculate grade distribution
        grade_dist = courses_df["Grade"].astype(str).str.upper().str.replace(r"[+-]", "", regex=True)
        grade_distribution = grade_dist.value_counts().to_dict()
        # Convert numpy types to native Python types
        grade_distribution = {str(k): int(v) for k, v in grade_distribution.items()}
        
        stats = {
            "total_courses": int(len(courses_df)),
            "average_gpa": float(valid_grades["GradePoint"].mean()) if len(valid_grades) > 0 else None,
            "grade_distribution": grade_distribution,
        }

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
        
        # Also save to skill_profile_parsed_single.csv (single student format)
        parsed_single_path = os.path.join("output", "skill_profile_parsed_single.csv")
        skills_df.to_csv(parsed_single_path, index=False)

        return {
            "student_id": student_id,
            "transcript_details": details,
            "num_courses_detected": int(len(courses_df)),
            "courses": courses_df.to_dict(orient="records"),
            "statistics": stats,
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

    # Load question bank early (used for validation fallback and related-skill expansion)
    try:
        qdf = load_question_bank()
        # Ensure numeric QuestionID
        qdf["QuestionID"] = pd.to_numeric(qdf["QuestionID"], errors="coerce")
        qdf = qdf.dropna(subset=["QuestionID"]).copy()
        qdf["QuestionID"] = qdf["QuestionID"].astype(int)
    except Exception:
        qdf = pd.DataFrame()

    # If per-student skill profile exists, normalize user-selected skill names
    # If a skill is not in the profile, check if it exists in the question bank as a fallback
    per_student_path = os.path.join("output", f"skill_profile_{student_id}.csv")
    profile_skills_map = {}
    if os.path.exists(per_student_path):
        try:
            skills_df_check = pd.read_csv(per_student_path)
            if "Skill" in skills_df_check.columns:
                # build canonical mapping from lowercased trimmed -> canonical value
                profile_skills_map = {str(x).strip().lower(): str(x).strip() for x in skills_df_check["Skill"].astype(str).tolist()}
        except Exception:
            # fallback to original flow if anything goes wrong reading the file
            pass

    # Build question bank skills map for fallback validation
    question_bank_skills_map = {}
    if not qdf.empty and "Skill" in qdf.columns:
        question_bank_skills_map = {str(x).strip().lower(): str(x).strip() for x in qdf["Skill"].astype(str).unique().tolist()}

    # Normalize selected skills: prefer profile canonical names, fallback to question bank canonical names
    normalized_selected = []
    invalid = []
    for s in selected:
        key = s.strip().lower()
        # First try to match against student's skill profile
        if profile_skills_map and key in profile_skills_map:
            normalized_selected.append(profile_skills_map[key])
        # Fallback to question bank if not in profile
        elif question_bank_skills_map and key in question_bank_skills_map:
            normalized_selected.append(question_bank_skills_map[key])
        else:
            invalid.append(s)

    if invalid:
        error_msg = f"Invalid skills selected (not found in student's skill profile or question bank): {invalid}"
        raise HTTPException(status_code=400, detail=error_msg)

    selected = normalized_selected  # use canonical names going forward

    # Expand selected skills with related skills if requested
    include_related = bool(payload.include_related)
    max_total = int(payload.max_total_skills or 5)
    if max_total <= 0:
        max_total = 5
    if max_total > 10:
        # sanity cap
        max_total = 10

    if include_related and not qdf.empty:
        expanded_skills, related_map = find_related_skills(selected, qdf, max_total=max_total)
    else:
        expanded_skills = selected[:max_total]
        related_map = {s: [] for s in selected}

    # Build pairs of (origin_selected_skill, skill_to_use)
    # origin is the skill the user selected; skill_to_use is the bank skill used to fetch/generate questions
    pairs: List[Tuple[str, str]] = []
    used_set = set()
    # First, add the original selected skills (if present in expanded)
    for s in selected:
        for e in expanded_skills:
            if e.lower() == s.lower() and e not in used_set:
                pairs.append((s, e))
                used_set.add(e)
                break

    # Then add remaining expanded skills and assign them to an origin (from related_map where possible)
    for e in expanded_skills:
        if e in used_set:
            continue
        # find origin that lists e
        origin = None
        for s, rels in related_map.items():
            if e in rels:
                origin = s
                break
        if origin is None:
            # fallback to first selected
            origin = selected[0]
        pairs.append((origin, e))
        used_set.add(e)
    
    # Create and save quiz plan records (one per selected skill)
    quiz_plan_records = []
    for skill in selected:
        # Try to get student's skill level from profile
        skill_level = "Unknown"
        if os.path.exists(per_student_path):
            try:
                skills_df_check = pd.read_csv(per_student_path)
                if "Skill" in skills_df_check.columns and "SkillLevel" in skills_df_check.columns:
                    skill_row = skills_df_check[skills_df_check["Skill"].astype(str).str.strip().str.lower() == skill.lower()]
                    if not skill_row.empty:
                        skill_level = str(skill_row.iloc[0]["SkillLevel"]).strip()
            except Exception:
                pass
        
        # Infer target difficulty from skill level (or use user-selected difficulty if "mixed")
        target_difficulty = payload.difficulty
        if target_difficulty == "mixed" or not target_difficulty:
            # Auto-infer based on skill level
            if skill_level in ["Beginner"]:
                target_difficulty = "Easy"
            elif skill_level in ["Developing"]:
                target_difficulty = "Medium"
            elif skill_level in ["Proficient", "Advanced"]:
                target_difficulty = "Hard"
            else:
                target_difficulty = "Medium"  # Default
        
        quiz_plan_records.append({
            "StudentID": student_id,
            "Skill": skill,
            "TargetDifficulty": target_difficulty,
            "NumQuestions": payload.num_questions_per_skill,
            "StudentSkillLevel": skill_level,
            "CreatedTime": datetime.now().isoformat(),
        })
    
    # Save quiz plan to CSV
    if quiz_plan_records:
        quiz_plan_df = pd.DataFrame(quiz_plan_records)
        quiz_plan_path = os.path.join("output", "quiz_plans_direct.csv")
        
        # Append to file if it exists, otherwise create new
        if os.path.exists(quiz_plan_path):
            try:
                existing_df = pd.read_csv(quiz_plan_path)
                # Optionally remove old plans for this student (keep only latest)
                existing_df = existing_df[existing_df["StudentID"] != student_id]
                combined_df = pd.concat([existing_df, quiz_plan_df], ignore_index=True)
                combined_df.to_csv(quiz_plan_path, index=False)
            except Exception:
                # If append fails, just overwrite
                quiz_plan_df.to_csv(quiz_plan_path, index=False)
        else:
            os.makedirs("output", exist_ok=True)
            quiz_plan_df.to_csv(quiz_plan_path, index=False)

    bank_skills = set(qdf["Skill"].astype(str).unique()) if not qdf.empty else set()
    aliases = load_skill_aliases() if "load_skill_aliases" in globals() else []

    questions = []
    answer_key = {}
    missing_origins = []

    next_qid = 100000  # ids for Gemini-generated questions

    # Iterate over pairs so we know the origin (user-selected) skill for each generated question
    for origin_skill, skill_to_use in pairs:
        mapped_skill = None
        if bank_skills:
            mapped_skill = map_to_bank_skill(skill_to_use, bank_skills, aliases)

        if not mapped_skill:
            mapped_skill = skill_to_use.split()[0].strip() if skill_to_use.split() else skill_to_use

        # --- Gemini + RAG ---
        gemini_ok = False
        try:
            context = load_skill_context(mapped_skill, kb_dir="knowledge_base", max_chars=5500)
            if context and context.strip():
                # Get retrieved chunk IDs if using RAG retrieval
                retrieved_chunk_ids = []
                try:
                    from src.rag_retrieval import retrieve_skill_context
                    # Try to load corpus and get chunk IDs
                    corpus_path = os.path.join("content", "skill_corpus.csv")
                    if os.path.exists(corpus_path):
                        corpus_df = pd.read_csv(corpus_path)
                        retrieval_result = retrieve_skill_context(
                            skill=mapped_skill,
                            corpus_df=corpus_df,
                            method="tfidf",
                            top_k=5
                        )
                        retrieved_chunk_ids = retrieval_result.get("retrieved_chunk_ids", [])
                except Exception:
                    # If RAG retrieval fails, continue without chunk IDs
                    retrieved_chunk_ids = []
                
                gen = generate_mcqs_from_context(
                    skill_key=mapped_skill,
                    context=context,
                    n=int(payload.num_questions_per_skill or 3),
                    difficulty=(payload.difficulty or "mixed").strip().lower(),
                    retrieved_chunk_ids=retrieved_chunk_ids if retrieved_chunk_ids else None
                )

                # Use validated questions (validation already done in generate_mcqs_from_context)
                gen_questions = gen.get("questions", []) if isinstance(gen, dict) else []
                
                # Log validation warnings if present (optional)
                if gen.get("warnings"):
                    # Could log to console or file
                    pass
                
                for q in gen_questions:
                    opts = q.get("options", {}) or {}
                    ans = str(q.get("answer", "")).strip().upper()

                    # Additional safety checks (redundant after validation but safe)
                    if ans not in ["A", "B", "C", "D"]:
                        continue
                    if not all(k in opts for k in ["A", "B", "C", "D"]):
                        continue
                    
                    # Check for duplicate options (additional safety)
                    opt_values = [str(opts.get(k, "")).strip().lower() for k in ["A", "B", "C", "D"]]
                    if len(opt_values) != len(set(opt_values)):
                        continue  # Skip if duplicates found

                    qid = next_qid
                    next_qid += 1
                    answer_key[str(qid)] = ans

                    questions.append(
                        {
                            "QuestionID": qid,
                            "SelectedSkill": origin_skill,
                            "Skill": mapped_skill,
                            "Difficulty": (payload.difficulty or "mixed").strip().lower(),
                            "QuestionText": str(q.get("question", "")).strip(),
                            "OptionA": str(opts.get("A", "")).strip(),
                            "OptionB": str(opts.get("B", "")).strip(),
                            "OptionC": str(opts.get("C", "")).strip(),
                            "OptionD": str(opts.get("D", "")).strip(),
                            "Explanation": str(q.get("explanation", "")).strip(),
                            "Source": "gemini",
                        }
                    )

                if len([x for x in questions if x["SelectedSkill"] == origin_skill]) > 0:
                    gemini_ok = True
        except Exception:
            gemini_ok = False

        # --- Fallback to Question Bank if Gemini failed ---
        if not gemini_ok:
            if qdf.empty:
                missing_origins.append(origin_skill)
                continue

            subset = qdf[qdf["Skill"].astype(str) == str(mapped_skill)].copy()
            difficulty = (payload.difficulty or "mixed").strip().lower()
            if difficulty != "mixed":
                subset = subset[subset["Difficulty"].astype(str).str.lower() == difficulty]

            if subset.empty:
                missing_origins.append(origin_skill)
                continue

            per_skill = int(payload.num_questions_per_skill or 3)
            subset = subset.sample(n=min(per_skill, len(subset)), random_state=42)

            for _, row in subset.iterrows():
                qid = int(row["QuestionID"])
                answer_key[str(qid)] = str(row["CorrectOption"]).strip().upper()

                questions.append(
                    {
                        "QuestionID": qid,
                        "SelectedSkill": origin_skill,
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

    if missing_origins:
        raise HTTPException(
            status_code=404,
            detail=f"No questions could be generated for selected origins: {sorted(set(missing_origins))}. Add knowledge_base files or expand question_bank.csv.",
        )

    if not questions:
        raise HTTPException(
            status_code=400,
            detail="No questions were generated. Please try different skills or check the knowledge base.",
        )
    
    # Final duplicate check across all questions (across all skills)
    try:
        from src.question_diversity import check_duplicate_questions
        is_diverse, duplicate_warnings, unique_questions = check_duplicate_questions(
            questions,
            similarity_threshold=0.85
        )
        if not is_diverse:
            # Use unique questions and log warnings (could be logged to file)
            questions = unique_questions
            # Note: warnings could be logged but not returned to user for simplicity
    except Exception:
        # If duplicate check fails, continue with original questions
        pass

    # Generate unique quiz_id
    quiz_id = str(uuid.uuid4())
    
    # Calculate time limit (default: 2 minutes per question, minimum 10 minutes)
    time_limit_minutes = max(10, len(questions) * 2)
    time_limit_seconds = time_limit_minutes * 60
    
    # Prepare questions for storage (with answers) and for frontend (without answers)
    questions_with_answers = []
    questions_for_frontend = []
    
    for q in questions:
        # Store full question with answer for backend
        questions_with_answers.append({
            "QuestionID": q["QuestionID"],
            "SelectedSkill": q.get("SelectedSkill", ""),
            "Skill": q.get("Skill", ""),
            "Difficulty": q.get("Difficulty", ""),
            "QuestionText": q.get("QuestionText", ""),
            "OptionA": q.get("OptionA", ""),
            "OptionB": q.get("OptionB", ""),
            "OptionC": q.get("OptionC", ""),
            "OptionD": q.get("OptionD", ""),
            "CorrectOption": answer_key.get(str(q["QuestionID"]), ""),
            "Explanation": q.get("Explanation", ""),
            "Source": q.get("Source", ""),
        })
        
        # Frontend version (no answer)
        questions_for_frontend.append({
            "QuestionID": q["QuestionID"],
            "SelectedSkill": q.get("SelectedSkill", ""),
            "Skill": q.get("Skill", ""),
            "Difficulty": q.get("Difficulty", ""),
            "QuestionText": q.get("QuestionText", ""),
            "OptionA": q.get("OptionA", ""),
            "OptionB": q.get("OptionB", ""),
            "OptionC": q.get("OptionC", ""),
            "OptionD": q.get("OptionD", ""),
            "Explanation": q.get("Explanation", ""),
            "Source": q.get("Source", ""),
        })
    
    # Generate quiz session token
    session_token = str(uuid.uuid4())
    start_time = datetime.now()
    
    # Save quiz metadata
    quiz_meta = {
        "quiz_id": quiz_id,
        "student_id": student_id,
        "selected_skills": selected,
        "expanded_skills": expanded_skills,
        "num_questions": len(questions),
        "num_questions_per_skill": payload.num_questions_per_skill,
        "difficulty": payload.difficulty,
        "time_limit_minutes": time_limit_minutes,
        "time_limit_seconds": time_limit_seconds,
        "created_time": start_time.isoformat(),
        "answer_key": answer_key,  # Store answer key in metadata
        "session_token": session_token,  # Session token for this quiz attempt
        "session_start_time": start_time.isoformat(),  # When quiz was started
    }
    
    # Save quiz files
    quiz_dir = os.path.join("output", "quizzes")
    os.makedirs(quiz_dir, exist_ok=True)
    
    questions_file = os.path.join(quiz_dir, f"{quiz_id}_questions.json")
    meta_file = os.path.join(quiz_dir, f"{quiz_id}_meta.json")
    
    with open(questions_file, "w", encoding="utf-8") as f:
        json.dump(questions_with_answers, f, indent=2, ensure_ascii=False)
    
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(quiz_meta, f, indent=2, ensure_ascii=False)
    
    # Also save answer key with student_id for backward compatibility
    save_quiz_answer_key(student_id, answer_key)
    
    # Return to frontend (without answers)
    return {
        "quiz_id": quiz_id,
        "student_id": student_id,
        "selected_skills": selected,
        "num_questions": len(questions),
        "time_limit_minutes": time_limit_minutes,
        "time_limit_seconds": time_limit_seconds,
        "session_token": session_token,  # Return session token
        "session_start_time": start_time.isoformat(),  # Return start time
        "questions": questions_for_frontend,  # No answers!
    }



@app.get("/quizzes/{quiz_id}")
def get_quiz(quiz_id: str):
    """
    Retrieve a quiz by quiz_id (for frontend to load quiz).
    Returns questions without answers.
    """
    meta_file = os.path.join("output", "quizzes", f"{quiz_id}_meta.json")
    questions_file = os.path.join("output", "quizzes", f"{quiz_id}_questions.json")
    
    if not os.path.exists(meta_file) or not os.path.exists(questions_file):
        raise HTTPException(status_code=404, detail=f"Quiz not found: {quiz_id}")
    
    # Load metadata
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    # Load questions (with answers)
    with open(questions_file, "r", encoding="utf-8") as f:
        questions_with_answers = json.load(f)
    
    # Remove answers for frontend
    questions_for_frontend = []
    for q in questions_with_answers:
        questions_for_frontend.append({
            "QuestionID": q["QuestionID"],
            "SelectedSkill": q.get("SelectedSkill", ""),
            "Skill": q.get("Skill", ""),
            "Difficulty": q.get("Difficulty", ""),
            "QuestionText": q.get("QuestionText", ""),
            "OptionA": q.get("OptionA", ""),
            "OptionB": q.get("OptionB", ""),
            "OptionC": q.get("OptionC", ""),
            "OptionD": q.get("OptionD", ""),
            "Explanation": q.get("Explanation", ""),
            "Source": q.get("Source", ""),
        })
    
    return {
        "quiz_id": quiz_id,
        "student_id": meta["student_id"],
        "selected_skills": meta["selected_skills"],
        "num_questions": meta["num_questions"],
        "time_limit_minutes": meta["time_limit_minutes"],
        "time_limit_seconds": meta["time_limit_seconds"],
        "created_time": meta["created_time"],
        "questions": questions_for_frontend,
    }


@app.post("/students/{student_id}/submit-quiz")
def submit_quiz(student_id: str, payload: SubmitQuizRequest):
    """
    Submit quiz responses with time limit enforcement.
    
    If quiz_id is provided, uses that quiz's answer key.
    Otherwise, falls back to student_id-based answer key (backward compatibility).
    
    Validates session token and time limit.
    """
    answer_key = {}
    meta = {}
    time_limit_seconds = None
    session_start_time = None
    
    # Try to load from quiz_id first (if provided in payload)
    if payload.quiz_id:
        meta_file = os.path.join("output", "quizzes", f"{payload.quiz_id}_meta.json")
        if os.path.exists(meta_file):
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
                answer_key = meta.get("answer_key", {})
                time_limit_seconds = meta.get("time_limit_seconds")
                session_start_time_str = meta.get("session_start_time")
                
                # Validate session token if provided
                if payload.session_token:
                    expected_token = meta.get("session_token")
                    if payload.session_token != expected_token:
                        raise HTTPException(
                            status_code=403,
                            detail="Invalid session token. Quiz session may have expired or been invalidated."
                        )
                
                # Validate time limit if session start time is available
                if session_start_time_str and time_limit_seconds:
                    try:
                        from datetime import datetime, timezone
                        session_start_time = datetime.fromisoformat(session_start_time_str)
                        current_time = datetime.now(timezone.utc) if session_start_time.tzinfo else datetime.now()
                        if session_start_time.tzinfo:
                            current_time = datetime.now(timezone.utc)
                        else:
                            current_time = datetime.now()
                        
                        elapsed_seconds = (current_time - session_start_time).total_seconds()
                        
                        if elapsed_seconds > time_limit_seconds:
                            # Time expired - reject submission
                            raise HTTPException(
                                status_code=400,
                                detail=f"Quiz time limit exceeded. Time limit: {time_limit_seconds}s, Elapsed: {elapsed_seconds:.1f}s"
                            )
                    except Exception as e:
                        if isinstance(e, HTTPException):
                            raise
                        # If time parsing fails, log but don't block (graceful degradation)
                        pass
                
                # Log violations if provided
                if payload.violations:
                    violations_file = os.path.join("output", "quizzes", f"{payload.quiz_id}_violations.json")
                    violations_data = {
                        "quiz_id": payload.quiz_id,
                        "student_id": student_id,
                        "violations": payload.violations,
                        "timestamp": datetime.now().isoformat(),
                    }
                    try:
                        if os.path.exists(violations_file):
                            with open(violations_file, "r", encoding="utf-8") as f:
                                existing = json.load(f)
                                if "violation_logs" not in existing:
                                    existing["violation_logs"] = []
                                existing["violation_logs"].append(violations_data)
                                violations_data = existing
                        with open(violations_file, "w", encoding="utf-8") as f:
                            json.dump(violations_data, f, indent=2, ensure_ascii=False)
                    except Exception:
                        # Log violations but don't fail submission
                        pass
    
    # Fallback to student_id-based answer key (backward compatibility)
    if not answer_key:
        key_path = os.path.join("output", f"quiz_answer_key_{student_id}.json")
        if not os.path.exists(key_path):
            raise HTTPException(status_code=404, detail="Answer key not found. Generate a quiz first.")
        
        with open(key_path, "r", encoding="utf-8") as f:
            answer_key = json.load(f)

    total = len(payload.responses)
    correct = 0
    per_question = []
    
    # Load questions to get skill information
    questions_file = None
    if payload.quiz_id:
        questions_file = os.path.join("output", "quizzes", f"{payload.quiz_id}_questions.json")
    
    questions_data = []
    if questions_file and os.path.exists(questions_file):
        with open(questions_file, "r", encoding="utf-8") as f:
            questions_data = json.load(f)
    
    # Create question lookup dict
    question_lookup = {str(q.get("QuestionID", "")): q for q in questions_data}

    # Score each response
    skill_stats = {}  # Track per-skill statistics
    
    for r in payload.responses:
        qid = str(r.question_id)
        gold = str(answer_key.get(qid, "")).upper()
        picked = str(r.selected_option).upper()
        # `r` is a Pydantic `QuizResponseItem` (object) not a dict; use attribute access
        response_time = getattr(r, "response_time_seconds", None)
        if response_time is None:
            response_time = 0.0

        is_correct = picked == gold and gold in ["A", "B", "C", "D"]
        if is_correct:
            correct += 1

        # Get question details
        question_info = question_lookup.get(qid, {})
        skill = question_info.get("Skill") or question_info.get("SelectedSkill") or "Unknown"
        question_text = question_info.get("QuestionText", "")
        difficulty = question_info.get("Difficulty", "")
        
        # Initialize skill stats if needed
        if skill not in skill_stats:
            skill_stats[skill] = {
                "total": 0,
                "correct": 0,
                "total_time": 0.0,
            }
        
        skill_stats[skill]["total"] += 1
        if is_correct:
            skill_stats[skill]["correct"] += 1
        skill_stats[skill]["total_time"] += response_time

        per_question.append(
            {
                "question_id": r.question_id,
                "question_text": question_text,
                "skill": skill,
                "difficulty": difficulty,
                "selected_option": picked,
                "correct_option": gold,
                "is_correct": is_correct,
                "response_time_seconds": response_time,
            }
        )
    
    # Calculate overall accuracy
    accuracy = (correct / total) if total else 0.0
    
    # Calculate per-skill accuracy
    per_skill_results = []
    for skill, stats in skill_stats.items():
        skill_accuracy = (stats["correct"] / stats["total"]) if stats["total"] > 0 else 0.0
        avg_time = (stats["total_time"] / stats["total"]) if stats["total"] > 0 else 0.0
        
        per_skill_results.append({
            "skill": skill,
            "total_questions": stats["total"],
            "correct_answers": stats["correct"],
            "accuracy": skill_accuracy,
            "avg_response_time_seconds": avg_time,
        })
    
    # Save detailed results to CSV
    results_dir = os.path.join("output", "quizzes")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save quiz_results_scored.csv (detailed results per question)
    detailed_results = []
    for pq in per_question:
        detailed_results.append({
            "StudentID": student_id,
            "QuizID": payload.quiz_id or "unknown",
            "QuestionID": pq["question_id"],
            "QuestionText": pq.get("question_text", ""),
            "Skill": pq.get("skill", ""),
            "Difficulty": pq.get("difficulty", ""),
            "SelectedOption": pq["selected_option"],
            "CorrectOption": pq["correct_option"],
            "IsCorrect": pq["is_correct"],
            "ResponseTimeSeconds": pq.get("response_time_seconds", 0.0),
        })
    
    if detailed_results:
        detailed_df = pd.DataFrame(detailed_results)
        results_scored_path = os.path.join("output", "quiz_results_scored.csv")
        
        # Append to file if exists, otherwise create new
        if os.path.exists(results_scored_path):
            try:
                existing_df = pd.read_csv(results_scored_path)
                combined_df = pd.concat([existing_df, detailed_df], ignore_index=True)
                combined_df.to_csv(results_scored_path, index=False)
            except Exception:
                detailed_df.to_csv(results_scored_path, index=False)
        else:
            detailed_df.to_csv(results_scored_path, index=False)
    
    # Save skill_quiz_updates.csv (per-skill aggregated results)
    if per_skill_results:
        skill_updates = []
        for skill_result in per_skill_results:
            skill_updates.append({
                "StudentID": student_id,
                "QuizID": payload.quiz_id or "unknown",
                "Skill": skill_result["skill"],
                "NumQuestions": skill_result["total_questions"],
                "NumCorrect": skill_result["correct_answers"],
                "QuizProficiency": skill_result["accuracy"],  # Proficiency = accuracy
                "AvgResponseTimeSeconds": skill_result["avg_response_time_seconds"],
            })
        
        skill_updates_df = pd.DataFrame(skill_updates)
        skill_updates_path = os.path.join("output", "skill_quiz_updates.csv")
        
        # Append to file if exists, otherwise create new
        if os.path.exists(skill_updates_path):
            try:
                existing_df = pd.read_csv(skill_updates_path)
                # Remove old entries for this student+quiz combination
                existing_df = existing_df[
                    ~((existing_df["StudentID"] == student_id) & 
                      (existing_df["QuizID"] == (payload.quiz_id or "unknown")))
                ]
                combined_df = pd.concat([existing_df, skill_updates_df], ignore_index=True)
                combined_df.to_csv(skill_updates_path, index=False)
            except Exception:
                skill_updates_df.to_csv(skill_updates_path, index=False)
        else:
            skill_updates_df.to_csv(skill_updates_path, index=False)
    
    # Trigger skill profile fusion after quiz scoring
    try:
        from src.skill_profile_fusion import fuse_profiles, load_skill_profiles, load_quiz_updates
        
        # Try to load skill profile (prefer parsed_single, fallback to explainable)
        parsed_single_path = os.path.join("output", "skill_profile_parsed_single.csv")
        explainable_path = os.path.join("output", "skill_profiles_explainable.csv")
        
        base_path = None
        if os.path.exists(parsed_single_path):
            base_path = parsed_single_path
        elif os.path.exists(explainable_path):
            base_path = explainable_path
        
        if base_path:
            try:
                base_df = load_skill_profiles(base_path)
                quiz_df = load_quiz_updates(skill_updates_path)
                fused_df = fuse_profiles(base_df, quiz_df)
                
                # Save fused profile
                fused_path = os.path.join("output", "skill_profiles_with_quiz.csv")
                fused_df.to_csv(fused_path, index=False)
            except Exception as e:
                # Log but don't fail quiz submission if fusion fails
                print(f"Warning: Skill fusion failed: {e}")
        else:
            print("Warning: No skill profile found for fusion. Skipping fusion step.")
    except ImportError:
        # If fusion module not available, skip silently
        pass
    except Exception as e:
        # Log but don't fail quiz submission
        print(f"Warning: Skill fusion error: {e}")

    return {
        "student_id": student_id,
        "quiz_id": payload.quiz_id,
        "total_answered": total,
        "correct": correct,
        "accuracy": accuracy,
        "per_question": per_question,
        "per_skill": per_skill_results,  # Add per-skill results
    }


# New endpoint: provide skills a UI can use to let user choose up to max_selectable skills.
@app.get("/students/{student_id}/selectable-skills")
def selectable_skills(student_id: str, top_n: int = 15, only_weak: bool = False):
    """
    Return a list of skills for the student suitable for selection in the quiz UI.

    - top_n: how many skills to return (default 15)
    - only_weak: if True, return only skills marked weak/missing or below score threshold
    """
    per_student_path = os.path.join("output", f"skill_profile_{student_id}.csv")
    path_primary = os.path.join("output", "skill_profiles_with_quiz.csv")
    path_fallback = os.path.join("output", "skill_profiles_explainable.csv")

    if os.path.exists(per_student_path):
        df = pd.read_csv(per_student_path)
        source = os.path.basename(per_student_path)
    else:
        path = path_primary if os.path.exists(path_primary) else path_fallback
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Skill profile CSV not found in output/")
        df = pd.read_csv(path)
        source = os.path.basename(path)

    # determine score column
    score_col = None
    for sc in ["ScoreNormalized", "FinalScore", "Score"]:
        if sc in df.columns:
            score_col = sc
            break

    # sort by score if available
    if score_col:
        df_sorted = df.sort_values(score_col, ascending=False).head(int(top_n))
    else:
        df_sorted = df.head(int(top_n))

    # determine weak flag using explainability column if present or a simple threshold
    has_weak_col = "IsWeakOrMissing" in df_sorted.columns
    out = []
    for _, r in df_sorted.iterrows():
        skill_name = str(r.get("Skill", r.get("skill", "")))
        # compute numeric score if possible
        score_val = None
        if score_col and pd.notna(r.get(score_col)):
            try:
                score_val = float(r.get(score_col))
                # normalize percentage-like FinalScore (> 1) to fraction if needed
                if score_col == "FinalScore" and score_val > 1:
                    # assume FinalScore is in 0-100, convert to 0-1 for threshold check but keep original number
                    pass
            except Exception:
                score_val = None

        is_weak = False
        if has_weak_col and pd.notna(r.get("IsWeakOrMissing")):
            try:
                is_weak = bool(r.get("IsWeakOrMissing"))
            except Exception:
                is_weak = str(r.get("IsWeakOrMissing")).strip().lower() in ("true", "1", "yes")
        else:
            # threshold-based weak detection
            if score_col and score_val is not None:
                if score_col == "FinalScore":
                    # treat FinalScore < 60 as weak (if FinalScore looks like percent)
                    is_weak = score_val < 60
                else:
                    # ScoreNormalized or other fractional scores: < 0.6 is weak
                    is_weak = score_val < 0.6

        if only_weak and not is_weak:
            continue

        out.append(
            {
                "skill": skill_name,
                "score": float(score_val) if score_val is not None else None,
                "level": str(r.get("FinalSkillLevel", r.get("SkillLevel", ""))),
                "is_weak": bool(is_weak),
                "selectable": True,
            }
        )

    return {
        "student_id": student_id,
        "source_file": source,
        "count": len(out),
        "max_selectable": 5,
        "skills": out,
    }


# New endpoint: return plain list of canonical skill values (easy for simple UI consumption)
@app.get("/students/{student_id}/available-skills")
def available_skills(student_id: str, only_weak: bool = False, limit: int = 100):
    """
    Return a simple list of canonical skill values for the student.
    - only_weak: if True, only return skills considered weak/missing
    - limit: maximum number of skills to return
    """
    per_student_path = os.path.join("output", f"skill_profile_{student_id}.csv")
    path_primary = os.path.join("output", "skill_profiles_with_quiz.csv")
    path_fallback = os.path.join("output", "skill_profiles_explainable.csv")

    if os.path.exists(per_student_path):
        df = pd.read_csv(per_student_path)
    else:
        path = path_primary if os.path.exists(path_primary) else path_fallback
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Skill profile CSV not found in output/")
        df = pd.read_csv(path)

    if "Skill" not in df.columns:
        raise HTTPException(status_code=500, detail="Skill column missing from skill profile file.")

    # compute weak flag similar to selectable_skills
    score_col = next((c for c in ["ScoreNormalized", "FinalScore", "Score"] if c in df.columns), None)
    has_weak_col = "IsWeakOrMissing" in df.columns

    out = []
    seen = set()
    for _, r in df.iterrows():
        val = str(r["Skill"]).strip()
        if not val or val in seen:
            continue

        is_weak = False
        if has_weak_col and pd.notna(r.get("IsWeakOrMissing")):
            try:
                is_weak = bool(r.get("IsWeakOrMissing"))
            except Exception:
                is_weak = str(r.get("IsWeakOrMissing")).strip().lower() in ("true", "1", "yes")
        elif score_col and pd.notna(r.get(score_col)):
            try:
                score_val = float(r.get(score_col))
                if score_col == "FinalScore":
                    is_weak = score_val < 60
                else:
                    is_weak = score_val < 0.6
            except Exception:
                is_weak = False

        if only_weak and not is_weak:
            continue

        out.append(val)
        seen.add(val)
        if len(out) >= int(limit):
            break

    return {"student_id": student_id, "count": len(out), "skills": out}
