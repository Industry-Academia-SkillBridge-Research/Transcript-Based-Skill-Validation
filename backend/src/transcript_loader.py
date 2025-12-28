import pandas as pd
from typing import Dict, List, Optional

# Mapping from SLIIT style letter grades to numeric points.
GRADE_POINTS: Dict[str, float] = {
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

# How many course slots each year has in the CSV
YEAR_CONFIG = {
    "Y1": {"year_num": 1, "max_courses": 9, "gpa_col": "Y1_GPA", "credits_col": "Y1_Credits"},
    "Y2": {"year_num": 2, "max_courses": 11, "gpa_col": "Y2_GPA", "credits_col": "Y2_Credits"},
    "Y3": {"year_num": 3, "max_courses": 9, "gpa_col": "Y3_GPA", "credits_col": "Y3_Credits"},
    "Y4": {"year_num": 4, "max_courses": 6, "gpa_col": "Y4_GPA", "credits_col": "Y4_Credits"},
}


def _read_csv_with_fallback(path: str) -> pd.DataFrame:
    """
    Read CSV trying a few encodings to avoid UnicodeDecodeError.
    """
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    # If all fail, let pandas raise its default error
    return pd.read_csv(path)


def load_transcripts_wide(path: str) -> pd.DataFrame:
    """
    Load the original 'wide' transcript file as provided by the university.
    Do not modify this format; use wide_to_long() to transform it.
    """
    df = _read_csv_with_fallback(path)

    # Basic cleaning for important identifier columns
    for col in ["Name", "RegNo", "Program", "Specialization", "Medium", "Admission"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df


def wide_to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wide transcript format into a long, analytics-friendly format.

    Output columns:
        StudentID, Name, Program, Specialization, Medium, Admission,
        Year, YearLabel,
        CourseCode, CourseTitle, Grade, GradePoint,
        YearGPA, YearCredits,
        WGPA, TotalCredits, ClassAward
    """
    records: List[Dict] = []

    for _, row in df_wide.iterrows():
        base_info = {
            "StudentID": str(row.get("RegNo", "")).strip(),
            "RegNo": str(row.get("RegNo", "")).strip(),
            "Name": str(row.get("Name", "")).strip(),
            "Program": str(row.get("Program", "")).strip(),
            "Specialization": str(row.get("Specialization", "")).strip(),
            "Medium": str(row.get("Medium", "")).strip(),
            "Admission": str(row.get("Admission", "")).strip(),
            "WGPA": row.get("WGPA", None),
            "TotalCredits": row.get("Total_Credits", None),
            "ClassAward": row.get("Class_Award", None),
        }

        for year_label, cfg in YEAR_CONFIG.items():
            year_num = cfg["year_num"]
            gpa_col = cfg["gpa_col"]
            credits_col = cfg["credits_col"]
            max_courses = cfg["max_courses"]

            year_gpa = row.get(gpa_col, None)
            year_credits = row.get(credits_col, None)

            for i in range(1, max_courses + 1):
                code_col = f"{year_label}_Code{i}"
                title_col = f"{year_label}_Title{i}"
                grade_col = f"{year_label}_Grade{i}"

                if code_col not in df_wide.columns:
                    # No such column in this dataset for this year
                    continue

                course_code = row.get(code_col, None)

                # Skip empty course slots
                if pd.isna(course_code) or str(course_code).strip() == "":
                    continue

                course_title = row.get(title_col, "")
                grade = row.get(grade_col, "")

                rec = dict(base_info)
                rec.update(
                    {
                        "Year": year_num,
                        "YearLabel": year_label,
                        "CourseCode": str(course_code).strip(),
                        "CourseTitle": str(course_title).strip(),
                        "Grade": str(grade).strip(),
                        "YearGPA": year_gpa,
                        "YearCredits": year_credits,
                    }
                )
                records.append(rec)

    df_long = pd.DataFrame(records)

    # Add grade points
    df_long["GradePoint"] = df_long["Grade"].map(GRADE_POINTS).fillna(0.0)

    return df_long


def load_transcripts_long(path: str) -> pd.DataFrame:
    """
    Convenience function: read the wide CSV and return the normalized long format.
    """
    df_wide = load_transcripts_wide(path)
    df_long = wide_to_long(df_wide)
    return df_long


def get_student_transcript(df_long: pd.DataFrame, student_id: str) -> pd.DataFrame:
    """
    Filter the long-format transcript DataFrame for a single student (by RegNo).
    """
    student_df = df_long[df_long["StudentID"] == student_id].copy()
    if student_df.empty:
        print(f"[WARN] No transcript rows found for StudentID={student_id}")
    return student_df


if __name__ == "__main__":
    # Quick manual test
    transcripts_long = load_transcripts_long("input/transcript_data.csv")
    print("Long-format rows:", len(transcripts_long))

    # Peek at the first few rows
    print(transcripts_long.head())

    # Try a sample student ID
    sample_id = transcripts_long["StudentID"].iloc[0]
    stu_df = get_student_transcript(transcripts_long, sample_id)
    print(f"\nTranscript for {sample_id}:")
    print(stu_df[["Year", "CourseCode", "CourseTitle", "Grade", "GradePoint"]])
