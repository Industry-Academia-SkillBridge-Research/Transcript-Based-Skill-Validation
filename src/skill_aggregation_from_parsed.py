import os
from typing import Dict, List

import pandas as pd

from course_skill_mapping import load_course_skill_mapping

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


def main():
    parsed_path = "output/transcript_parsed_single.csv"
    mapping_path = "input/course_skill_mapping.csv"
    out_path = "output/skill_profile_parsed_single.csv"

    print(f"Loading parsed transcript from: {parsed_path}")
    tdf = load_parsed_transcript(parsed_path)
    print(f"Parsed transcript rows (after cleaning): {len(tdf)}")

    print(f"Loading course-skill mapping from: {mapping_path}")
    mapping_dict = load_course_skill_mapping(mapping_path)
    mdf = mapping_to_dataframe(mapping_dict)
    print(f"Mapping rows (course-skill): {len(mdf)}")

    skill_profile = aggregate_skills_from_parsed(tdf, mdf)
    print(f"Skill rows: {len(skill_profile)}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    skill_profile.to_csv(out_path, index=False)
    print(f"Saved skill profile to: {out_path}")

    # Quick sample
    sid = skill_profile["StudentID"].iloc[0]
    print(f"\nSample skills for {sid}:")
    print(
        skill_profile[skill_profile["StudentID"] == sid]
        .sort_values("ScoreNormalized", ascending=False)
        .head(10)
    )


if __name__ == "__main__":
    main()
