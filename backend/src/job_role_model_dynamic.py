import os
from typing import List, Tuple

import pandas as pd


# Prefer quiz-aware profile if it exists
SKILL_PROFILE_FUSED = "output/skill_profiles_with_quiz.csv"
SKILL_PROFILE_BASE = "output/skill_profiles_explainable.csv"
ROLE_TEMPLATES_PATH = "output/job_role_skill_templates_dynamic.csv"
SUMMARY_OUT = "output/role_readiness_dynamic.csv"
DETAILS_OUT = "output/role_readiness_explainable.csv"


def load_skill_profiles() -> pd.DataFrame:
    """
    Load student-skill profiles.

    Priority:
      1) output/skill_profiles_with_quiz.csv  (transcript + quiz)
      2) output/skill_profiles_explainable.csv  (transcript only)
    """
    if os.path.exists(SKILL_PROFILE_FUSED):
        path = SKILL_PROFILE_FUSED
        print(f"Loading fused (transcript + quiz) skill profiles from: {path}")
    else:
        path = SKILL_PROFILE_BASE
        print(f"[WARN] Fused skill profiles not found. Falling back to baseline: {path}")

    df = pd.read_csv(path)

    # We now expect:
    #  - ScoreNormalized  (baseline from transcript)
    #  - FinalScore       (fused transcript + quiz)  [if fused file]
    #  - FinalSkillLevel  (if fused file)
    # For simplicity, we will use:
    #  - if FinalScore exists -> use that as SkillScore
    #  - else -> SkillScore = ScoreNormalized
    if "FinalScore" in df.columns:
        df["SkillScore"] = df["FinalScore"]
        df["SkillLevelEffective"] = df.get("FinalSkillLevel", "Unknown")
    else:
        df["SkillScore"] = df["ScoreNormalized"]
        df["SkillLevelEffective"] = df.get("SkillLevel", "Unknown")

    required = {"StudentID", "Skill", "SkillScore"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Skill profile file missing columns: {missing}")

    return df


def load_role_templates(path: str = ROLE_TEMPLATES_PATH) -> pd.DataFrame:
    """
    Load dynamic role skill templates from job postings.

    Expected columns:
      RoleName, Skill, ImportanceNorm (and others created in job_postings_ingestion)
    """
    df = pd.read_csv(path)
    if "ImportanceNorm" not in df.columns:
        df["ImportanceNorm"] = 1.0
    return df


def compute_role_readiness(
    profiles_df: pd.DataFrame,
    templates_df: pd.DataFrame,
    weak_threshold: float = 0.4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each student and each role, compute readiness using dynamic templates.

    Returns:
      - readiness_summary_df: one row per (StudentID, RoleName)
      - readiness_details_df: one row per (StudentID, RoleName, Skill)
    """
    if profiles_df.empty or templates_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    students = profiles_df["StudentID"].unique()
    roles = templates_df["RoleName"].unique()

    summary_records: List[dict] = []
    detail_records: List[dict] = []

    # index by (StudentID, Skill) for fast lookup
    profiles_indexed = profiles_df.set_index(["StudentID", "Skill"])

    for student_id in students:
        for role in roles:
            role_skills = templates_df[templates_df["RoleName"] == role]
            if role_skills.empty:
                continue

            total_importance = role_skills["ImportanceNorm"].sum()
            if total_importance == 0:
                continue

            attained_weighted = 0.0
            weak_or_missing: List[str] = []
            num_skills = len(role_skills)
            num_present = 0

            for _, row in role_skills.iterrows():
                skill = row["Skill"]
                importance = row["ImportanceNorm"]

                student_score = 0.0
                student_level = "None"

                if (student_id, skill) in profiles_indexed.index:
                    srow = profiles_indexed.loc[(student_id, skill)]
                    student_score = float(srow["SkillScore"])
                    student_level = str(srow["SkillLevelEffective"])
                    num_present += 1

                attained_fraction = student_score  # both [0,1]
                attained_weighted += importance * attained_fraction

                is_weak = student_score < weak_threshold
                if is_weak:
                    weak_or_missing.append(skill)

                detail_records.append(
                    {
                        "StudentID": student_id,
                        "RoleName": role,
                        "Skill": skill,
                        "RequiredImportance": importance,
                        "StudentScore": student_score,
                        "StudentLevel": student_level,
                        "AttainedFraction": attained_fraction,
                        "IsWeakOrMissing": is_weak,
                    }
                )

            readiness_score = attained_weighted / total_importance
            coverage = num_present / num_skills

            summary_records.append(
                {
                    "StudentID": student_id,
                    "RoleName": role,
                    "ReadinessScore": readiness_score,
                    "Coverage": coverage,
                    "NumSkills": num_skills,
                    "NumSkillsPresent": num_present,
                    "NumWeakOrMissing": len(weak_or_missing),
                    "WeakOrMissingSkills": ", ".join(weak_or_missing[:15]),
                }
            )

    readiness_summary_df = pd.DataFrame(summary_records)
    readiness_details_df = pd.DataFrame(detail_records)
    return readiness_summary_df, readiness_details_df


def main():
    profiles_df = load_skill_profiles()
    print(f"Loaded {len(profiles_df)} student-skill rows")

    templates_df = load_role_templates()
    print(f"Loaded {len(templates_df)} role-skill template rows")

    summary_df, details_df = compute_role_readiness(profiles_df, templates_df)

    print(f"Role readiness summary rows: {len(summary_df)}")
    print(f"Role readiness detailed rows: {len(details_df)}")

    os.makedirs("output", exist_ok=True)
    summary_df.to_csv(SUMMARY_OUT, index=False)
    details_df.to_csv(DETAILS_OUT, index=False)

    # Print sample for one student
    if not summary_df.empty:
        example_student = summary_df["StudentID"].iloc[0]
        print(f"\nReadiness for {example_student}:")
        print(
            summary_df[summary_df["StudentID"] == example_student]
            .sort_values("ReadinessScore", ascending=False)
            .head()
        )

        print(f"\nDetails for top role of {example_student}:")
        top_role = (
            summary_df[summary_df["StudentID"] == example_student]
            .sort_values("ReadinessScore", ascending=False)["RoleName"]
            .iloc[0]
        )
        print(
            details_df[
                (details_df["StudentID"] == example_student)
                & (details_df["RoleName"] == top_role)
            ][
                [
                    "Skill",
                    "RequiredImportance",
                    "StudentScore",
                    "StudentLevel",
                    "AttainedFraction",
                    "IsWeakOrMissing",
                ]
            ].head(15)
        )


if __name__ == "__main__":
    main()
