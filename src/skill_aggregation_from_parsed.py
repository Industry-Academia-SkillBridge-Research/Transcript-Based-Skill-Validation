import os
from typing import Dict, List, Optional

import pandas as pd

from src.course_skill_mapping import load_course_skill_mapping

# Letter grade → numeric points
GRADE_TO_POINTS: Dict[str, float] = {
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
    "": 0.0,
}

YEAR_WEIGHTS = {
    1: 0.8,
    2: 1.0,
    3: 1.0,
    4: 1.2,
}


def load_parsed_transcript(path: str) -> pd.DataFrame:
    """
    Load the CSV produced by transcript_ingestion.py.

    Expected columns:
      StudentID, RegNo, CourseCode, CourseTitle, Grade, Year
    """
    df = pd.read_csv(path)

    # Clean up
    df = df[df["CourseCode"].notna()]

    # Normalize grade text
    df["Grade"] = (
        df["Grade"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"NAN": ""})
    )

    # Drop rows with no grade at all
    df = df[df["Grade"] != ""]

    # Map to numeric
    df["GradePoint"] = df["Grade"].map(GRADE_TO_POINTS).fillna(0.0)

    # Year weights (default 1.0 if missing)
    def _year_weight(y):
        try:
            y_int = int(y)
            return YEAR_WEIGHTS.get(y_int, 1.0)
        except Exception:
            return 1.0

    df["YearWeight"] = df["Year"].apply(_year_weight)

    # Overall contribution of this course
    df["CourseContribution"] = df["GradePoint"] / 4.0 * df["YearWeight"]

    return df


def mapping_to_dataframe(mapping: Dict[str, dict]) -> pd.DataFrame:
    """
    Convert the course_skill_mapping dict into a DataFrame with one row per skill.
    """
    rows: List[dict] = []
    for code, info in mapping.items():
        for skill in info["skills"]:
            rows.append(
                {
                    "CourseCode": code,
                    "Skill": skill,
                    "MainSkill": info["main_skill"],
                    "CurriculumSkillLevel": info["skill_level"],
                }
            )
    return pd.DataFrame(rows)


def aggregate_skills_from_parsed(
    transcript_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join parsed transcript with course→skill mapping and aggregate to skill profile.

    Returns columns:
      StudentID, Skill, EvidenceCount, TotalContribution,
      ScoreNormalized, SkillLevel
    """
    merged = transcript_df.merge(mapping_df, on="CourseCode", how="left")

    # Keep only rows where we actually know the skills
    merged = merged[merged["Skill"].notna()].copy()

    if merged.empty:
        raise ValueError("No mapped skills found for the parsed transcript.")

    # Aggregate
    group_cols = ["StudentID", "Skill"]
    agg = (
        merged.groupby(group_cols)
        .agg(
            EvidenceCount=("CourseCode", "nunique"),
            TotalContribution=("CourseContribution", "sum"),
        )
        .reset_index()
    )

    # Normalize per student
    max_per_student = (
        agg.groupby("StudentID")["TotalContribution"].transform("max").replace(0, 1e-6)
    )
    agg["ScoreNormalized"] = agg["TotalContribution"] / max_per_student

    # Map score to level
    def score_to_level(s: float) -> str:
        if s >= 0.7:
            return "Advanced"
        elif s >= 0.4:
            return "Developing"
        else:
            return "Beginner"

    agg["SkillLevel"] = agg["ScoreNormalized"].apply(score_to_level)

    return agg

def build_skill_profile_from_parsed(
    parsed_df: pd.DataFrame,
    mapping_path: str = "input/course_skill_mapping.csv",
) -> pd.DataFrame:
    """
    Take a parsed transcript (one student) and build a skill profile.

    This is basically your current script logic, but parameterised.
    Expected columns in parsed_df: StudentID, CourseCode, Grade, (optionally Year)
    """
    if parsed_df.empty:
        return pd.DataFrame()

    # Load course → skill mapping
    mapping_df = pd.read_csv(mapping_path)

    # Melt mapping_df so each (CourseCode, Skill) is a row
    skill_cols = ["Skill1", "Skill2", "Skill3", "Skill4", "Skill5"]
    mapping_long = []
    for _, row in mapping_df.iterrows():
        code = str(row["CourseCode"]).strip()
        level = str(row.get("SkillLevel", "")).strip()
        for col in skill_cols:
            skill = row.get(col)
            if isinstance(skill, str) and skill.strip():
                mapping_long.append(
                    {
                        "CourseCode": code,
                        "Skill": skill.strip(),
                        "SkillLevelTemplate": level,
                    }
                )

    mapping_long = pd.DataFrame(mapping_long)
    if mapping_long.empty:
        return pd.DataFrame()

    # Join parsed transcript with mapping
    parsed_df["CourseCode"] = parsed_df["CourseCode"].astype(str).str.strip()
    merged = parsed_df.merge(mapping_long, on="CourseCode", how="left")
    merged = merged.dropna(subset=["Skill"])

    # Map grades to numeric weights (simplified, adjust if needed)
    grade_to_weight = {
        "A+": 1.0,
        "A": 1.0,
        "A-": 0.95,
        "B+": 0.85,
        "B": 0.8,
        "B-": 0.75,
        "C+": 0.65,
        "C": 0.6,
        "C-": 0.55,
        "D+": 0.45,
        "D": 0.4,
        "E": 0.2,
        "F": 0.0,
        "": 0.0,
        None: 0.0,
    }

    merged["Grade"] = merged["Grade"].fillna("").astype(str).str.strip()
    merged["GradeWeight"] = merged["Grade"].map(grade_to_weight).fillna(0.0)

    # For now, YearWeight = 1 (you can reuse your year weighting if you had it)
    merged["YearWeight"] = 1.0
    merged["Contribution"] = merged["GradeWeight"] * merged["YearWeight"]

    # Aggregate by student + skill
    student_id = parsed_df["StudentID"].iloc[0]
    grouped = (
        merged.groupby(["StudentID", "Skill"], as_index=False)
        .agg(
            EvidenceCount=("CourseCode", "nunique"),
            TotalContribution=("Contribution", "sum"),
        )
    )

    # Normalise contribution to [0,1] for ScoreNormalized
    max_contrib = grouped["TotalContribution"].max()
    if max_contrib > 0:
        grouped["ScoreNormalized"] = grouped["TotalContribution"] / max_contrib
    else:
        grouped["ScoreNormalized"] = 0.0

    # Simple level bucketing
    def level_from_score(s: float) -> str:
        if s >= 0.75:
            return "Advanced"
        if s >= 0.5:
            return "Developing"
        if s > 0:
            return "Beginner"
        return "None"

    grouped["SkillLevel"] = grouped["ScoreNormalized"].apply(level_from_score)

    return grouped

def main():
    parsed_path = "output/transcript_parsed_single.csv"
    mapping_path = "input/course_skill_mapping.csv"
    out_path = "output/skill_profile_parsed_single.csv"

    print(f"Loading parsed transcript from: {parsed_path}")
    parsed_df = pd.read_csv(parsed_path)
    # Basic cleaning if needed
    parsed_df = parsed_df.dropna(subset=["CourseCode"])

    print(f"Parsed transcript rows (after cleaning): {len(parsed_df)}")
    print(f"Loading course-skill mapping from: {mapping_path}")
    print(f"Mapping rows (course-skill): (see mapping file)")

    skill_df = build_skill_profile_from_parsed(parsed_df, mapping_path)

    print(f"Skill rows: {len(skill_df)}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    skill_df.to_csv(out_path, index=False)
    print(f"Saved skill profile to: {out_path}")
    print("\nSample skills for", parsed_df['StudentID'].iloc[0], ":")
    print(skill_df.head(10))


if __name__ == "__main__":
    main()
