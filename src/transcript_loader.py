import pandas as pd
from typing import List, Dict, Any


def load_transcripts(path: str = "input/transcript_data.csv") -> pd.DataFrame:
    """
    Load the transcript CSV into a pandas DataFrame.

    This is the raw academic dataset that we will convert into skill features.
    """
    # same encoding trick, in case transcripts have odd characters
    df = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"[INFO] Loaded transcripts with encoding: {enc}")
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        df = pd.read_csv(path)

    return df


def get_student_row(df: pd.DataFrame, reg_no: str) -> pd.Series:
    """
    Return the row for a single student identified by RegNo.
    """
    matches = df[df["RegNo"] == reg_no]
    if matches.empty:
        raise ValueError(f"No student found with RegNo={reg_no}")
    # assume unique RegNo
    return matches.iloc[0]


def extract_student_courses(student_row: pd.Series) -> List[Dict[str, Any]]:
    """
    Extract all course entries (code, title, grade, year) for one student.

    Returns a list like:
        [
          {"year": 1, "code": "IT1010", "title": "Introduction to Programming", "grade": "C+"},
          {"year": 1, "code": "IT1020", ...},
          ...
        ]
    """
    courses: List[Dict[str, Any]] = []

    # you have years 1..4, and up to ~11 subjects per year in your sample
    for year in range(1, 5):
        for idx in range(1, 15):  # generous upper bound; missing cols are skipped
            code_col = f"Y{year}_Code{idx}"
            title_col = f"Y{year}_Title{idx}"
            grade_col = f"Y{year}_Grade{idx}"

            # skip if this column set does not exist in the CSV
            if code_col not in student_row.index:
                continue

            code = student_row[code_col]

            # stop if no course code in this slot
            if pd.isna(code) or str(code).strip() == "":
                continue

            title = student_row[title_col] if title_col in student_row.index else ""
            grade = student_row[grade_col] if grade_col in student_row.index else ""

            courses.append(
                {
                    "year": year,
                    "code": str(code).strip(),
                    "title": str(title).strip(),
                    "grade": str(grade).strip(),
                }
            )

    return courses


if __name__ == "__main__":
    # quick manual test with your sample RegNo
    df = load_transcripts("input/transcript_data.csv")

    # replace with an actual RegNo from your CSV, for example:
    reg_no = "IT21709618"  # Lauren Scott in your sample

    student_row = get_student_row(df, reg_no)
    courses = extract_student_courses(student_row)

    print(f"Found {len(courses)} courses for {reg_no}")
    for c in courses[:10]:  # print first 10
        print(c)
