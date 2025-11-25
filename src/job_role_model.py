import json
from typing import Dict, List
import pandas as pd


def load_skill_profiles(path: str) -> pd.DataFrame:
    """
    Load the skill_profiles_baseline.csv produced by skill_aggregation.py.
    """
    df = pd.read_csv(path)
    # Clean Skill names just in case
    df["Skill"] = df["Skill"].astype(str).str.strip()
    return df


def load_job_role_templates(path: str) -> List[dict]:
    """
    Load job role templates from JSON.

    Each role:
      {
        "role_id": "...",
        "role_name": "...",
        "skills": [
          {"name": "...", "weight": 1.0},
          ...
        ]
      }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def compute_readiness_for_student(
    student_skills: pd.DataFrame,
    role_templates: List[dict],
    missing_threshold: float = 0.4,
) -> pd.DataFrame:
    """
    Compute readiness scores for one student across all roles.

    student_skills: subset of skill_profiles for a single StudentID.
    Returns a DataFrame with:
      RoleID, RoleName, ReadinessScore, Coverage, MissingSkills
    """
    results = []

    # Build quick lookup: skill_name -> normalized score
    skill_score_map: Dict[str, float] = {
        row["Skill"]: float(row["ScoreNormalized"])
        for _, row in student_skills.iterrows()
    }

    for role in role_templates:
        role_id = role["role_id"]
        role_name = role["role_name"]
        required_skills = role["skills"]

        total_weight = 0.0
        weighted_score_sum = 0.0
        skills_covered = 0
        missing_skills: List[str] = []

        for item in required_skills:
            skill_name = item["name"]
            weight = float(item.get("weight", 1.0))
            total_weight += weight

            score = skill_score_map.get(skill_name, 0.0)
            weighted_score_sum += score * weight

            if skill_name in skill_score_map and score > 0:
                skills_covered += 1
            if score < missing_threshold:
                missing_skills.append(skill_name)

        if total_weight == 0:
            readiness = 0.0
        else:
            # readiness in [0, 1], then scale to percentage
            readiness = weighted_score_sum / total_weight

        coverage = skills_covered / max(len(required_skills), 1)

        results.append(
            {
                "RoleID": role_id,
                "RoleName": role_name,
                "ReadinessScore": round(readiness * 100, 2),  # percentage
                "Coverage": round(coverage * 100, 2),         # percentage of skills with some evidence
                "MissingSkills": ", ".join(missing_skills),
            }
        )

    return pd.DataFrame(results)


def compute_readiness_for_all_students(
    skill_profiles: pd.DataFrame,
    role_templates: List[dict],
) -> pd.DataFrame:
    """
    Compute readiness scores for all students.

    Returns a DataFrame where each row is (StudentID, RoleID, ...).
    """
    records: List[dict] = []

    for student_id, stu_df in skill_profiles.groupby("StudentID"):
        name = stu_df["Name"].iloc[0]
        program = stu_df["Program"].iloc[0]
        specialization = stu_df["Specialization"].iloc[0]

        df_roles = compute_readiness_for_student(stu_df, role_templates)

        for _, row in df_roles.iterrows():
            rec = {
                "StudentID": student_id,
                "Name": name,
                "Program": program,
                "Specialization": specialization,
                "RoleID": row["RoleID"],
                "RoleName": row["RoleName"],
                "ReadinessScore": row["ReadinessScore"],
                "Coverage": row["Coverage"],
                "MissingSkills": row["MissingSkills"],
            }
            records.append(rec)

    return pd.DataFrame(records)


if __name__ == "__main__":
    # Load baseline skill profiles
    skills_df = load_skill_profiles("output/skill_profiles_baseline.csv")

    # Load job role templates
    role_templates = load_job_role_templates("configs/job_role_templates.json")

    # Compute readiness for all students
    readiness_df = compute_readiness_for_all_students(skills_df, role_templates)
    print("Readiness rows:", len(readiness_df))

    # Show a sample for one student
    sample_id = skills_df["StudentID"].iloc[0]
    sample_roles = readiness_df[readiness_df["StudentID"] == sample_id]
    print(f"\nRole readiness for {sample_id}:")
    print(
        sample_roles[
            ["RoleName", "ReadinessScore", "Coverage", "MissingSkills"]
        ].sort_values(by="ReadinessScore", ascending=False)
    )

    # Save to CSV
    readiness_df.to_csv("output/role_readiness_baseline.csv", index=False)
    print("\nSaved to output/role_readiness_baseline.csv")
