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

    return {
        "candidate_name": grab(r"NAME OF CANDIDATE\s*[:\-]?\s*(.+)"),
        "programme": grab(r"PROGRAMME\s*[:\-]?\s*(.+)"),
        "specialization": grab(r"FIELD OF SPECIALIZATION\s*[:\-]?\s*(.+)"),
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
