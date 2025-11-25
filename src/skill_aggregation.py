from typing import Dict, List
import os
from pathlib import Path

import pandas as pd

from course_skill_mapping import load_course_skill_mapping
from transcript_loader import load_transcripts_long, get_student_transcript

# Year weights: later years count more for skill mastery
YEAR_WEIGHTS: Dict[int, float] = {
    1: 1.0,
    2: 1.2,
    3: 1.5,
    4: 1.7,
}

# Skill level weights from course_skill_mapping.csv
SKILL_LEVEL_WEIGHTS: Dict[str, float] = {
    "Beginner": 0.8,
    "Intermediate": 1.0,
    "Advanced": 1.2,
    # Mixed levels
    "Beginner/Intermediate": 0.9,
    "Intermediate/Advanced": 1.1,
}


def _get_year_weight(year: int) -> float:
    """Return a numeric weight for the academic year."""
    try:
        year_int = int(year)
    except (TypeError, ValueError):
        return 1.0
    return YEAR_WEIGHTS.get(year_int, 1.0)


def _get_skill_level_weight(level: str) -> float:
    """
    Map SkillLevel text (e.g. 'Beginner', 'Beginner/Intermediate') to a numeric weight.

    Handles simple variants like spaces and short forms.
    """
    if not isinstance(level, str):
        return 1.0

    # Normalize spacing and case
    level_norm = level.replace(" ", "").lower()

    if level_norm in {"beginner", "beg"}:
        key = "Beginner"
    elif level_norm in {"intermediate", "int"}:
        key = "Intermediate"
    elif level_norm in {"advanced", "adv"}:
        key = "Advanced"
    elif level_norm in {"beginner/intermediate", "beg/int"}:
        key = "Beginner/Intermediate"
    elif level_norm in {"intermediate/advanced", "int/adv"}:
        key = "Intermediate/Advanced"
    else:
        # Fallback: try original string
        key = level.strip()

    return SKILL_LEVEL_WEIGHTS.get(key, 1.0)


def map_score_to_level(score_normalized: float) -> str:
    """
    Map a normalized score (0–1) to a categorical skill level.
    """
    if score_normalized >= 0.85:
        return "Advanced"
    elif score_normalized >= 0.65:
        return "Proficient"
    elif score_normalized >= 0.45:
        return "Developing"
    else:
        return "Beginner"


def aggregate_skills_for_all_students(
    transcripts_long: pd.DataFrame,
    course_mapping: Dict[str, dict],
) -> pd.DataFrame:
    """
    Aggregate course-level information into skill-level scores for all students.

    Inputs:
      transcripts_long: output of load_transcripts_long()
      course_mapping: output of load_course_skill_mapping()

    Output columns:
      StudentID, Name, Program, Specialization,
      Skill, EvidenceCount, ScoreRaw, ScorePerCourse,
      ScoreNormalized, SkillLevel
    """
    records: List[Dict] = []

    # Group by student
    for student_id, stu_df in transcripts_long.groupby("StudentID"):
        name = stu_df["Name"].iloc[0]
        program = stu_df["Program"].iloc[0]
        specialization = stu_df["Specialization"].iloc[0]

        # skill -> list of numeric contributions
        skill_contribs: Dict[str, List[float]] = {}

        for _, row in stu_df.iterrows():
            course_code = str(row["CourseCode"]).strip()
            grade_point = float(row["GradePoint"])
            year = int(row["Year"])

            if course_code not in course_mapping:
                # course not mapped yet
                continue

            course_info = course_mapping[course_code]
            skills = course_info.get("skills", [])
            main_skill = course_info.get("main_skill", "")
            course_level = course_info.get("skill_level", "")

            if not skills:
                continue

            year_w = _get_year_weight(year)
            level_w = _get_skill_level_weight(course_level)

            for skill in skills:
                contrib = grade_point * year_w * level_w

                # Slight boost if this is the main skill of the course
                if skill == main_skill:
                    contrib *= 1.2

                skill_contribs.setdefault(skill, []).append(contrib)

        # Convert skill_contribs into output records
        for skill, contrib_list in skill_contribs.items():
            score_raw = sum(contrib_list)
            evidence_count = len(contrib_list)
            score_per_course = score_raw / evidence_count if evidence_count > 0 else 0.0

            # Normalize to 0–1 assuming 4.0 is max grade
            score_normalized = min(score_per_course / 4.0, 1.0)
            skill_level = map_score_to_level(score_normalized)

            records.append(
                {
                    "StudentID": student_id,
                    "Name": name,
                    "Program": program,
                    "Specialization": specialization,
                    "Skill": skill,
                    "EvidenceCount": evidence_count,
                    "ScoreRaw": round(score_raw, 3),
                    "ScorePerCourse": round(score_per_course, 3),
                    "ScoreNormalized": round(score_normalized, 3),
                    "SkillLevel": skill_level,
                }
            )

    df_skills = pd.DataFrame(records)
    return df_skills


if __name__ == "__main__":
    # Load inputs
    transcripts_long = load_transcripts_long("input/transcript_data.csv")
    course_mapping = load_course_skill_mapping("input/course_skill_mapping.csv")

    # Run aggregation
    df_skills = aggregate_skills_for_all_students(transcripts_long, course_mapping)
    print("Generated skill rows:", len(df_skills))

    # Show a sample for one student
    sample_id = transcripts_long["StudentID"].iloc[0]
    sample_skills = df_skills[df_skills["StudentID"] == sample_id]
    print(f"\nSkill profile for {sample_id}:")
    print(
        sample_skills[
            ["Skill", "EvidenceCount", "ScorePerCourse", "ScoreNormalized", "SkillLevel"]
        ].sort_values(by="ScoreNormalized", ascending=False)
    )

    # Ensure output folder exists
    Path("output").mkdir(parents=True, exist_ok=True)

    # Save to CSV for analysis and later ML steps
    out_path = os.path.join("output", "skill_profiles_baseline.csv")
    df_skills.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
