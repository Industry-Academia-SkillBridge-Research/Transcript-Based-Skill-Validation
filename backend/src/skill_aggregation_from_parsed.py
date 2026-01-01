# src/skill_aggregation_from_parsed.py
from typing import Dict, List, Optional

import pandas as pd

from src.course_skill_mapping import load_course_skill_mapping

GRADE_POINTS = {
    "A+": 4.0, "A": 4.0, "A-": 3.7,
    "B+": 3.3, "B": 3.0, "B-": 2.7,
    "C+": 2.3, "C": 2.0, "C-": 1.7,
    "D+": 1.3, "D": 1.0, "D-": 0.7,
    "E": 0.0, "F": 0.0,
}

# Year weights: later years weighted more (final year courses are more important)
YEAR_WEIGHTS = {
    1: 0.8,   # Foundation courses
    2: 1.0,   # Core courses (baseline)
    3: 1.1,   # Advanced courses
    4: 1.2,   # Specialization courses (highest weight)
}

# Maximum possible score = max grade point (4.0) * max year weight (1.2) = 4.8
MAX_COURSE_SCORE = 4.0 * max(YEAR_WEIGHTS.values())  # 4.8


def year_weight(year: Optional[int]) -> float:
    """Return importance weight for a given year (1-4)."""
    if year is None or pd.isna(year):
        return 1.0  # Default weight if year unknown
    year_int = int(year)
    return YEAR_WEIGHTS.get(year_int, 1.0)


def build_skill_profile_from_parsed(
    student_id: str,
    parsed_courses_df: pd.DataFrame,
    mapping_path: str = "input/course_skill_mapping.csv",
) -> pd.DataFrame:
    """
    Convert course rows into skill profile with year-weighted scores.
    
    Input df must contain: CourseCode, CourseTitle, Grade
    Optional: Year (if not present, will try to infer from CourseCode)
    Output: StudentID, Skill, EvidenceCount, ScoreNormalized, SkillLevel
    
    Score calculation:
    - Grade point (0-4) × Year weight (0.8-1.2) = Contribution
    - Normalized to [0, 1] using MAX_COURSE_SCORE (4.8)
    - Aggregated per skill (average contribution)
    """
    mapping = load_course_skill_mapping(mapping_path)

    # Prepare dataframe
    df = parsed_courses_df.copy()
    df["CourseCode"] = df["CourseCode"].astype(str).str.strip()
    df["Grade"] = df["Grade"].astype(str).str.strip()
    df["GradePoint"] = df["Grade"].map(GRADE_POINTS).fillna(0.0)
    
    # Ensure Year column exists (infer if missing)
    if "Year" not in df.columns or df["Year"].isna().all():
        def infer_year_from_code(code: str) -> Optional[int]:
            if not code or len(code) < 4:
                return None
            try:
                if code.upper().startswith("IT") and code[2].isdigit():
                    year = int(code[2])
                    if 1 <= year <= 4:
                        return year
            except (ValueError, IndexError):
                pass
            return None
        
        df["Year"] = df["CourseCode"].apply(infer_year_from_code)

    # Aggregate skill contributions with year weighting
    skill_rows: List[Dict] = []
    for _, row in df.iterrows():
        code = row["CourseCode"]
        if code not in mapping:
            continue

        skills = mapping[code]["skills"]
        gp = float(row["GradePoint"])
        year = row.get("Year")
        yw = year_weight(year)
        
        # Calculate contribution: grade point × year weight
        # This gives more weight to final year courses
        contribution_base = gp * yw
        
        # Distribute contribution evenly across all skills for this course
        if not skills:
            continue
        
        per_skill_contribution = contribution_base / len(skills)

        for skill in skills:
            skill_rows.append({
                "Skill": skill,
                "Contribution": per_skill_contribution
            })

    if not skill_rows:
        return pd.DataFrame(columns=["StudentID", "Skill", "EvidenceCount", "ScoreNormalized", "SkillLevel"])

    # Aggregate per skill
    sdf = pd.DataFrame(skill_rows)
    agg = sdf.groupby("Skill").agg(
        EvidenceCount=("Contribution", "count"),  # Number of courses contributing
        TotalContribution=("Contribution", "sum"),  # Sum of contributions
    ).reset_index()
    
    # Normalize to [0, 1] using max possible score
    # If a skill has multiple courses, average the contributions
    # But we normalize using max single course score (4.8) so scores can exceed 1.0 for skills with many courses
    # Clip to [0, 1] to keep scores in range
    agg["ScoreNormalized"] = (agg["TotalContribution"] / agg["EvidenceCount"]) / MAX_COURSE_SCORE
    agg["ScoreNormalized"] = agg["ScoreNormalized"].clip(0.0, 1.0)

    # Assign skill levels based on normalized score
    def level(x: float) -> str:
        if x >= 0.75:
            return "Advanced"
        elif x >= 0.50:
            return "Proficient"
        elif x >= 0.25:
            return "Developing"
        else:
            return "Beginner"

    agg["SkillLevel"] = agg["ScoreNormalized"].apply(level)
    agg.insert(0, "StudentID", student_id)
    
    # Select and order columns
    return agg[["StudentID", "Skill", "EvidenceCount", "ScoreNormalized", "SkillLevel"]]
