import os
import re
import argparse
from typing import Optional, List, Dict

import pdfplumber
from PIL import Image
import pytesseract
import pandas as pd


# -----------------------------
# Text extraction
# -----------------------------

def extract_text_from_pdf(path: str) -> str:
    """Extract plain text from a PDF transcript using pdfplumber."""
    if pdfplumber is None:
        raise ImportError("pdfplumber is not installed. Please `pip install pdfplumber`.")

    text_chunks: List[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_chunks.append(page_text)

    return "\n".join(text_chunks)


def extract_text_from_image(path: str, tesseract_cmd: Optional[str] = None) -> str:
    """Extract plain text from an image transcript using Tesseract OCR."""
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    img = Image.open(path)
    text = pytesseract.image_to_string(img)
    return text


def extract_text_from_file(path: str, tesseract_cmd: Optional[str] = None) -> str:
    """Dispatch extraction based on file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext in [".pdf"]:
        return extract_text_from_pdf(path)
    elif ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
        return extract_text_from_image(path, tesseract_cmd=tesseract_cmd)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    

# -----------------------------
# Parsing transcript text
# -----------------------------

GRADE_PATTERN = re.compile(r"\b(A\+?|A-|B\+?|B-|C\+?|C-|D\+?|D|E|F)\b")
COURSE_CODE_PATTERN = re.compile(r"(IT\d{4})", re.IGNORECASE)


def parse_transcript_text(
    text: str,
    student_id: Optional[str] = None,
    regno: Optional[str] = None,
) -> pd.DataFrame:
    """
    Very simple parser that looks for lines containing ITxxxx course codes
    and a grade token at the end of the line.

    Returns DataFrame with columns:
    StudentID, RegNo, CourseCode, CourseTitle, Grade, Year
    """
    rows: List[Dict] = []
    current_year: Optional[int] = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Try to capture "Year 1", "Year II", etc. (optional)
        m_year = re.search(r"Year\s+(\d+)", line, flags=re.IGNORECASE)
        if m_year:
            try:
                current_year = int(m_year.group(1))
            except ValueError:
                current_year = None
            continue

        # Look for course code like IT1010
        m_code = COURSE_CODE_PATTERN.search(line)
        if not m_code:
            continue

        code = m_code.group(1).upper()

        # Text after the code we treat as title + grade + maybe other columns
        tail = line[m_code.end() :].strip()

        # Try to detect grade as last grade-like token
        tokens = tail.split()
        grade = ""
        title = tail

        if tokens:
            last_token = tokens[-1]
            m_grade = GRADE_PATTERN.fullmatch(last_token)
            if m_grade:
                grade = m_grade.group(1)
                title = " ".join(tokens[:-1]).strip().rstrip(",;:-")

        if not title:
            title = tail

        rows.append(
            {
                "StudentID": student_id or "UNKNOWN",
                "RegNo": regno or "",
                "CourseCode": code,
                "CourseTitle": title,
                "Grade": grade,
                "Year": current_year,
            }
        )

    df = pd.DataFrame(rows)
    print(f"Parsed {len(df)} course rows.")
    return df

def parse_transcript_file(
    file_path: str,
    student_id: Optional[str] = None,
    regno: Optional[str] = None,
    tesseract_cmd: Optional[str] = None,
) -> pd.DataFrame:
    """
    High-level helper used by both CLI and FastAPI.

    1) Extract text from a PDF/image.
    2) Parse courses/grades.
    3) Attach StudentID / RegNo if provided.
    """
    text = extract_text_from_file(file_path, tesseract_cmd=tesseract_cmd)
    df = parse_transcript_text(text)

    # Ensure required columns exist
    if "StudentID" not in df.columns:
        df["StudentID"] = None
    if "RegNo" not in df.columns:
        df["RegNo"] = None

    if student_id:
        df["StudentID"] = df["StudentID"].fillna(student_id).replace("", student_id)
    if regno:
        df["RegNo"] = df["RegNo"].fillna(regno).replace("", regno)

    return df


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file", required=True,
        help="Path to transcript file (PDF or image).",
    )
    parser.add_argument(
        "--tesseract-cmd", default=None,
        help="Optional full path to tesseract.exe if not on PATH.",
    )
    parser.add_argument(
        "--out-csv", default="output/transcript_parsed_single.csv",
        help="Where to save the parsed CSV.",
    )
    parser.add_argument(
        "--student-id", default=None,
        help="Student ID to attach to parsed rows.",
    )
    parser.add_argument(
        "--regno", default=None,
        help="Registration number to attach to parsed rows.",
    )

    args = parser.parse_args()

    print(f"Reading transcript from: {args.file}")
    text = extract_text_from_file(args.file, tesseract_cmd=args.tesseract_cmd)

    print("Parsing transcript text...")
    df = parse_transcript_text(
        text,
        student_id=args.student_id,
        regno=args.regno,
    )

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    print(f"Saved to: {args.out_csv}")
    print(df.head())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to transcript PDF/image")
    parser.add_argument("--tesseract-cmd", help="Optional full path to tesseract.exe")
    parser.add_argument("--out-csv", default="output/transcript_parsed_single.csv")
    parser.add_argument("--student-id", help="Student ID (optional)")
    parser.add_argument("--regno", help="Registration number (optional)")
    args = parser.parse_args()

    print(f"Reading transcript from: {args.file}")
    df = parse_transcript_file(
        file_path=args.file,
        student_id=args.student_id,
        regno=args.regno,
        tesseract_cmd=args.tesseract_cmd,
    )

    print(f"Parsed {len(df)} course rows.")
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved to: {args.out_csv}")
    print(df.head())

