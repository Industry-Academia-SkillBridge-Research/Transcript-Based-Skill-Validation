# src/transcript_ingestion.py
import re
from typing import Dict, Tuple

import pandas as pd

# ---------- transcript parsing helpers ----------

COURSE_LINE_RE = re.compile(
    r"\b(IT\d{4})\b\s+(.+?)\s+([A-F][+-]?)\b"
)

def extract_transcript_details(text: str) -> Dict[str, str]:
    """
    Best-effort extraction. Different transcript templates will vary.
    Adjust regex patterns once you see more samples.
    """
    def grab(pattern: str) -> str:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        return m.group(1).strip() if m else ""

    # Try multiple patterns for student ID/registration number
    student_id_patterns = [
        r"REGISTRATION\s*(?:NUMBER|NO\.?)\s*[:\-]?\s*(IT\d{8})",  # REGISTRATION NUMBER: IT21001288
        r"REG\.?\s*NO\.?\s*[:\-]?\s*(IT\d{8})",  # REG. NO: IT21001288
        r"STUDENT\s*ID(?:ENTIFICATION)?\s*(?:NUMBER)?\s*[:\-]?\s*(IT\d{8})",  # STUDENT ID: IT21001288
        r"(IT\d{8})",  # Just look for IT followed by 8 digits (common format)
    ]
    
    student_id = ""
    for pattern in student_id_patterns:
        student_id = grab(pattern)
        if student_id:
            break
    
    return {
        "candidate_name": grab(r"NAME OF CANDIDATE\s*[:\-]?\s*(.+)"),
        "programme": grab(r"PROGRAMME\s*[:\-]?\s*(.+)"),
        "specialization": grab(r"FIELD OF SPECIALIZATION\s*[:\-]?\s*(.+)"),
        "student_id": student_id,
        "registration_number": student_id,  # Same as student_id for compatibility
    }

def parse_transcript_text(text: str) -> Tuple[Dict[str, str], pd.DataFrame]:
    """
    Returns:
      - details dict (name/program/specialization if found)
      - courses dataframe: CourseCode, CourseTitle, Grade
    """
    details = extract_transcript_details(text)

    courses = []
    for line in text.splitlines():
        line = " ".join(line.split())
        m = COURSE_LINE_RE.search(line)
        if not m:
            continue
        code, title, grade = m.group(1), m.group(2).strip(), m.group(3).strip()
        courses.append({"CourseCode": code, "CourseTitle": title, "Grade": grade})

    df = pd.DataFrame(courses).drop_duplicates(subset=["CourseCode", "CourseTitle", "Grade"])
    return details, df
