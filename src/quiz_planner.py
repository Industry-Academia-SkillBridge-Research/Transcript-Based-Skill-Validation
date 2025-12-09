import pandas as pd
from typing import List

# -----------------------------
# Config / hyper-parameters
# -----------------------------

# Below this fraction, we treat a skill as weak and a candidate for quizzing.
# AttainedFraction is in [0, 1], where 1 = fully meets role requirement.
WEAK_THRESHOLD = 0.6

# How many skills to include per (student, role) quiz session
MAX_SKILLS_PER_STUDENT_ROLE = 5

# How many questions to ask per skill (you can tune this later)
QUESTIONS_PER_SKILL = 3

# Map current mastery level → difficulty of quiz questions
DIFFICULTY_MAP = {
    "Beginner": "Easy",
    "Developing": "Medium",
    "Proficient": "Hard",
    "Advanced": "Hard",
}


def load_role_readiness_details(path: str) -> pd.DataFrame:
    """
    Load the detailed role readiness file.

    Expected columns (from job_role_model_dynamic):
        StudentID, RoleName, Skill, RequiredImportance,
        StudentScore, StudentLevel, AttainedFraction, IsWeakOrMissing
    """
    df = pd.read_csv(path)
    # Basic cleaning
    df["StudentID"] = df["StudentID"].astype(str).str.strip()
    df["RoleName"] = df["RoleName"].astype(str).str.strip()
    df["Skill"] = df["Skill"].astype(str).str.strip()
    return df


def infer_difficulty(student_level: str) -> str:
    """
    Map the student's current skill level to a target quiz difficulty.
    """
    level = str(student_level).strip()
    return DIFFICULTY_MAP.get(level, "Medium")


def plan_quiz_for_group(group: pd.DataFrame) -> pd.DataFrame:
    """
    Plan a quiz for a single (StudentID, RoleName) group.

    Steps:
      1. Filter weak skills using AttainedFraction < WEAK_THRESHOLD
      2. Sort by RequiredImportance (desc) then AttainedFraction (asc)
      3. Pick top N skills
      4. Assign difficulty and number of questions
    """
    if group.empty:
        return pd.DataFrame()

    # Ensure required columns exist
    required_cols = [
        "StudentID",
        "RoleName",
        "Skill",
        "RequiredImportance",
        "StudentScore",
        "StudentLevel",
        "AttainedFraction",
        "IsWeakOrMissing",
    ]
    for col in required_cols:
        if col not in group.columns:
            raise ValueError(f"Missing column in role readiness details: {col}")

    # 1) Filter weak / missing skills (based on AttainedFraction and/or IsWeakOrMissing)
    weak_mask = (group["AttainedFraction"] < WEAK_THRESHOLD) | (group["IsWeakOrMissing"].astype(bool))
    weak_skills = group[weak_mask].copy()

    if weak_skills.empty:
        # No weak skills → no quiz needed for this role
        return pd.DataFrame()

    # 2) Sort: most important skills first, lowest mastery first
    weak_skills = weak_skills.sort_values(
        by=["RequiredImportance", "AttainedFraction"],
        ascending=[False, True],
    )

    # 3) Pick top N skills for this student-role pair
    weak_skills = weak_skills.head(MAX_SKILLS_PER_STUDENT_ROLE)

    # 4) Build quiz plan rows
    plan_rows = []
    for _, row in weak_skills.iterrows():
        student_id = row["StudentID"]
        role_name = row["RoleName"]
        skill = row["Skill"]
        required_importance = float(row["RequiredImportance"])
        student_score = float(row["StudentScore"])
        attained_fraction = float(row["AttainedFraction"])
        level = str(row["StudentLevel"])
        difficulty = infer_difficulty(level)

        plan_rows.append(
            {
                "StudentID": student_id,
                "RoleName": role_name,
                "Skill": skill,
                "StudentLevel": level,
                "RequiredImportance": required_importance,
                "StudentScore": student_score,
                "AttainedFraction": attained_fraction,
                "TargetDifficulty": difficulty,
                "NumQuestions": QUESTIONS_PER_SKILL,
            }
        )

    return pd.DataFrame(plan_rows)


def build_quiz_plans(
    details_path: str = "output/role_readiness_details_dynamic.csv",
    output_path: str = "output/quiz_plans.csv",
) -> pd.DataFrame:
    """
    Build quiz plans for all (StudentID, RoleName) pairs
    and save them as a CSV.
    """
    details_df = load_role_readiness_details(details_path)

    # Group by StudentID + RoleName, then plan quiz per group
    grouped = details_df.groupby(["StudentID", "RoleName"], group_keys=False)
    plans_list: List[pd.DataFrame] = []

    for (student_id, role_name), group in grouped:
        plan_df = plan_quiz_for_group(group)
        if not plan_df.empty:
            plans_list.append(plan_df)

    if not plans_list:
        print("No quiz plans generated (no weak skills found).")
        return pd.DataFrame()

    all_plans = pd.concat(plans_list, ignore_index=True)

    # Save to CSV
    all_plans.to_csv(output_path, index=False)
    print(f"Quiz plans saved to {output_path}")
    print(f"Total quiz rows: {len(all_plans)}")

    return all_plans


if __name__ == "__main__":
    plans = build_quiz_plans()

    # Show a small sample for manual inspection
    print("\nSample quiz plans:")
    print(plans.head(10))
