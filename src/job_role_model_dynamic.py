import os
from typing import List

import pandas as pd


def load_skill_profiles(path: str) -> pd.DataFrame:
    """
    Load student-skill profiles from explainable aggregation.
    """
    df = pd.read_csv(path)
    # expected columns: StudentID, Skill, EvidenceCount, TotalContribution, ScoreNormalized, SkillLevel
    return df


def load_role_templates(path: str) -> pd.DataFrame:
    """
    Load dynamic role skill templates from job postings.
    """
    df = pd.read_csv(path)
    # expected columns: RoleName, Skill, JobCount, RolePostingCount, Support, Importance, ImportanceNorm
    return df


def compute_role_readiness(
    profiles_df: pd.DataFrame,
    templates_df: pd.DataFrame,
    weak_threshold: float = 0.4,
) -> (pd.DataFrame, pd.DataFrame):
    """
    For each student and each role, compute readiness using dynamic templates.

    Returns:
      - readiness_summary_df: one row per (StudentID, RoleName)
      - readiness_details_df: one row per (StudentID, RoleName, Skill)
    """
    # Prepare
    if profiles_df.empty or templates_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Make sure ImportanceNorm exists
    if "ImportanceNorm" not in templates_df.columns:
        templates_df["ImportanceNorm"] = 1.0

    # All students and roles
    students = profiles_df["StudentID"].unique()
    roles = templates_df["RoleName"].unique()

    summary_records = []
    detail_records = []

    # Index skill profiles by (StudentID, Skill) for faster lookup
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

            # per-skill details
            for _, row in role_skills.iterrows():
                skill = row["Skill"]
                importance = row["ImportanceNorm"]

                # default: student has 0 for this skill
                student_score = 0.0
                student_level = "None"

                if (student_id, skill) in profiles_indexed.index:
                    student_row = profiles_indexed.loc[(student_id, skill)]
                    student_score = float(student_row["ScoreNormalized"])
                    student_level = str(student_row["SkillLevel"])
                    num_present += 1

                # fraction of required importance that student meets
                attained_fraction = student_score  # since both in [0,1]

                attained_weighted += importance * attained_fraction

                if student_score < weak_threshold:
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
                        "IsWeakOrMissing": student_score < weak_threshold,
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
    profiles_df = load_skill_profiles("output/skill_profiles_explainable.csv")
    templates_df = load_role_templates("output/job_role_skill_templates_dynamic.csv")

    print(f"Loaded {len(profiles_df)} student-skill rows")
    print(f"Loaded {len(templates_df)} role-skill template rows")

    summary_df, details_df = compute_role_readiness(profiles_df, templates_df)

    print(f"Role readiness summary rows: {len(summary_df)}")
    print(f"Role readiness detailed rows: {len(details_df)}")

    os.makedirs("output", exist_ok=True)
    summary_df.to_csv("output/role_readiness_dynamic.csv", index=False)
    details_df.to_csv("output/role_readiness_details_dynamic.csv", index=False)


    # Print sample for one student
    if not summary_df.empty:
        example_student = summary_df["StudentID"].iloc[0]
        print(f"\nReadiness for {example_student}:")
        print(summary_df[summary_df["StudentID"] == example_student].sort_values("ReadinessScore", ascending=False).head())

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
