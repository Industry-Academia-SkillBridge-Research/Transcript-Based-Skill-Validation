import os
from typing import Dict, List

import pandas as pd

from course_skill_mapping import load_course_skill_mapping
from transcript_loader import load_transcripts_long  

# 1) Grade → numeric points 
GRADE_POINTS = {
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

# 2) Year weights (later years slightly more important)
YEAR_WEIGHTS = {
    1: 0.8,   # foundation
    2: 1.0,   # core
    3: 1.1,   # advanced
    4: 1.2,   # specialization
}

MAX_COURSE_SCORE = 4.0 * max(YEAR_WEIGHTS.values())  # 4.8

def grade_to_points(grade: str) -> float:
    """Convert letter grade to numeric points."""
    return GRADE_POINTS.get(str(grade).strip(), 0.0)


def year_weight(year: int) -> float:
    """Return importance weight for a given year (1-4)."""
    return YEAR_WEIGHTS.get(int(year), 1.0)


def build_course_skill_rows(
    long_df: pd.DataFrame,
    course_mapping: Dict[str, dict],
) -> pd.DataFrame:
    """
    Expand transcript long-format rows into per-course-per-skill rows.

    Each row represents:
      StudentID, CourseCode, Year, Grade, GradePoint,
      Skill, CourseWeight, Contribution
    """
    records: List[dict] = []

    for _, row in long_df.iterrows():
        student_id = row["StudentID"]
        course_code = str(row["CourseCode"]).strip()
        course_title = row["CourseTitle"]
        grade = str(row["Grade"]).strip()
        year = int(row["Year"])

        gp = grade_to_points(grade)
        yw = year_weight(year)

        # Overall "importance" of this course -> skill
        contribution_base = gp * yw

        mapping = course_mapping.get(course_code)
        if not mapping:
            # course not in mapping table
            continue

        # Distribute contribution across all skills for this course
        skills = mapping["skills"]
        if not skills:
            continue

        per_skill_contribution = contribution_base / len(skills)

        for skill in skills:
            records.append(
                {
                    "StudentID": student_id,
                    "CourseCode": course_code,
                    "CourseTitle": course_title,
                    "Year": year,
                    "Grade": grade,
                    "GradePoint": gp,
                    "YearWeight": yw,
                    "Skill": skill,
                    "CourseContribution": per_skill_contribution,
                }
            )

    return pd.DataFrame(records)

def aggregate_skill_scores(course_skill_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate course-level contributions to a final skill score per student.

    Returns one row per (StudentID, Skill) with:
      EvidenceCount, TotalContribution, ScoreNormalized, SkillLevel
    """
    if course_skill_df.empty:
        return pd.DataFrame()

    grouped = (
        course_skill_df.groupby(["StudentID", "Skill"])
        .agg(
            EvidenceCount=("CourseCode", "nunique"),
            TotalContribution=("CourseContribution", "sum"),
        )
        .reset_index()
    )

    # Normalize contribution to [0, 1] using a global max score
    grouped["ScoreNormalized"] = grouped["TotalContribution"] / MAX_COURSE_SCORE
    grouped["ScoreNormalized"] = grouped["ScoreNormalized"].clip(0.0, 1.0)

    # Assign levels (you can tune thresholds)
    def level(score: float) -> str:
        if score >= 0.75:
            return "Advanced"
        elif score >= 0.5:
            return "Proficient"
        elif score >= 0.25:
            return "Developing"
        else:
            return "Beginner"

    grouped["SkillLevel"] = grouped["ScoreNormalized"].apply(level)

    return grouped

def build_explanations(
    course_skill_df: pd.DataFrame,
    skill_scores_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a detailed explanation table:

    One row per (StudentID, Skill, CourseCode) with:
      Contribution, ShareOfSkillContribution, etc.
    """
    if course_skill_df.empty or skill_scores_df.empty:
        return pd.DataFrame()

    # Merge to get total contribution per (StudentID, Skill)
    merged = course_skill_df.merge(
        skill_scores_df[
            ["StudentID", "Skill", "TotalContribution", "ScoreNormalized", "SkillLevel"]
        ],
        on=["StudentID", "Skill"],
        how="left",
    )

    # Share of skill contribution
    merged["ShareOfSkillContribution"] = (
        merged["CourseContribution"] / merged["TotalContribution"]
    ).fillna(0.0)

    return merged


def main():
    # 1) Load transcript long-format
    long_df = load_transcripts_long("input/transcript_data.csv")
    print(f"Long-format transcript rows: {len(long_df)}")

    # 2) Load course → skill mapping
    course_mapping = load_course_skill_mapping("input/course_skill_mapping.csv")
    print(f"Loaded {len(course_mapping)} course mappings")

    # 3) Build per-course-per-skill rows
    course_skill_df = build_course_skill_rows(long_df, course_mapping)
    print(f"Per-course-per-skill rows: {len(course_skill_df)}")

    # 4) Aggregate to skill scores
    skill_scores_df = aggregate_skill_scores(course_skill_df)
    print(f"Student-skill rows: {len(skill_scores_df)}")

    # 5) Build explanation table
    explanations_df = build_explanations(course_skill_df, skill_scores_df)
    print(f"Explanation rows: {len(explanations_df)}")

    # 6) Save outputs
    os.makedirs("output", exist_ok=True)
    skill_scores_df.to_csv("output/skill_profiles_explainable.csv", index=False)
    explanations_df.to_csv("output/skill_explanations.csv", index=False)

    # 7) Print one example for your viva
    example_student = long_df["StudentID"].iloc[0]
    print(f"\nSkill scores for {example_student}:")
    print(skill_scores_df[skill_scores_df["StudentID"] == example_student].head())

    print(f"\nExplanation for one skill of {example_student}:")
    if not skill_scores_df.empty:
        example_skill = (
            skill_scores_df[skill_scores_df["StudentID"] == example_student]["Skill"]
            .iloc[0]
        )
        expl = explanations_df[
            (explanations_df["StudentID"] == example_student)
            & (explanations_df["Skill"] == example_skill)
        ]
        print(
            expl[
                [
                    "Skill",
                    "CourseCode",
                    "CourseTitle",
                    "Grade",
                    "GradePoint",
                    "Year",
                    "YearWeight",
                    "CourseContribution",
                    "ShareOfSkillContribution",
                ]
            ]
        )


if __name__ == "__main__":
    main()
