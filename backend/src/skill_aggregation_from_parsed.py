# src/skill_aggregation_from_parsed.py
from typing import Dict, List

import pandas as pd

from src.course_skill_mapping import load_course_skill_mapping

GRADE_POINTS = {
    "A+": 4.0, "A": 4.0, "A-": 3.7,
    "B+": 3.3, "B": 3.0, "B-": 2.7,
    "C+": 2.3, "C": 2.0, "C-": 1.7,
    "D+": 1.3, "D": 1.0, "E": 0.0, "F": 0.0,
}

def build_skill_profile_from_parsed(
    student_id: str,
    parsed_courses_df: pd.DataFrame,
    mapping_path: str = "input/course_skill_mapping.csv",
) -> pd.DataFrame:
    """
    Input df must contain: CourseCode, CourseTitle, Grade
    Output: StudentID, Skill, EvidenceCount, ScoreNormalized, SkillLevel
    """
    mapping = load_course_skill_mapping(mapping_path)

    # course -> gradepoint
    df = parsed_courses_df.copy()
    df["CourseCode"] = df["CourseCode"].astype(str).str.strip()
    df["Grade"] = df["Grade"].astype(str).str.strip()
    df["GradePoint"] = df["Grade"].map(GRADE_POINTS).fillna(0.0)

    # aggregate skill contributions
    skill_rows: List[Dict] = []
    for _, row in df.iterrows():
        code = row["CourseCode"]
        if code not in mapping:
            continue

        skills = mapping[code]["skills"]
        gp = float(row["GradePoint"])

        # simple normalization: 0..4 -> 0..1
        score = gp / 4.0

        for s in skills:
            skill_rows.append({"Skill": s, "Contribution": score})

    if not skill_rows:
        return pd.DataFrame(columns=["StudentID","Skill","EvidenceCount","ScoreNormalized","SkillLevel"])

    sdf = pd.DataFrame(skill_rows)
    agg = sdf.groupby("Skill").agg(
        EvidenceCount=("Contribution","count"),
        ScoreNormalized=("Contribution","mean"),
    ).reset_index()

    def level(x: float) -> str:
        if x >= 0.75:
            return "Advanced"
        if x >= 0.45:
            return "Developing"
        return "Beginner"

    agg["SkillLevel"] = agg["ScoreNormalized"].apply(level)
    agg.insert(0, "StudentID", student_id)
    return agg
