from typing import Dict, List
import numpy as np

from course_skill_mapping import load_course_skill_mapping
from transcript_loader import (
    load_transcripts,
    get_student_row,
    extract_student_courses,
)


def grade_to_weight(grade: str) -> float:
    """
    Convert letter grade to a numeric weight in [0, 1].

    This is a simple encoding that can be tuned later.
    """
    grade = grade.strip().upper()

    mapping = {
        "A+": 1.00,
        "A": 1.00,
        "A-": 0.90,
        "B+": 0.85,
        "B": 0.80,
        "B-": 0.75,
        "C+": 0.70,
        "C": 0.65,
        "C-": 0.60,
        "D+": 0.55,
        "D": 0.50,
        "E": 0.30,
        "F": 0.20,
        "FAIL": 0.20,
        "PASS": 0.65,  # in case of pass/fail modules
    }

    return mapping.get(grade, 0.0)  # unknown/missing grade → 0.0


def skill_level_factor(skill_level: str) -> float:
    """
    Give a small boost for intermediate/advanced modules.

    Beginner:       1.0
    Intermediate:   1.05
    Advanced:       1.10
    """
    level = skill_level.strip().lower()

    if "advanced" in level:
        return 1.10
    if "intermediate" in level:
        return 1.05
    if "beginner" in level:
        return 1.00
    return 1.00


def compute_skill_profile(
    student_courses: List[Dict],
    course_mapping: Dict[str, dict],
) -> Dict[str, Dict[str, float]]:
    """
    Compute aggregated skill scores for one student.

    Returns:
        {
          "Programming Fundamentals & C Language": {
              "score": 0.78,
              "n_courses": 3
          },
          ...
        }
    """

    skill_scores: Dict[str, List[float]] = {}

    for course in student_courses:
        code = course["code"]
        grade = course["grade"]

        if code not in course_mapping:
            # course not in our mapping → skip for now
            continue

        course_info = course_mapping[code]
        main_skill = course_info["main_skill"]
        level = course_info["skill_level"]

        g_weight = grade_to_weight(grade)
        level_factor = skill_level_factor(level)

        final_weight = g_weight * level_factor

        if main_skill not in skill_scores:
            skill_scores[main_skill] = []

        skill_scores[main_skill].append(final_weight)

    # aggregate with mean
    aggregated: Dict[str, Dict[str, float]] = {}
    for skill, weights in skill_scores.items():
        aggregated[skill] = {
            "score": float(np.mean(weights)),
            "n_courses": len(weights),
        }

    return aggregated


if __name__ == "__main__":
    # 1. Load data
    mapping = load_course_skill_mapping("input/course_skill_mapping.csv")
    df = load_transcripts("input/transcript_data.csv")

    # 2. Choose a student to test
    reg_no = "IT21709618"  # change to any valid RegNo in your CSV

    student_row = get_student_row(df, reg_no)
    student_courses = extract_student_courses(student_row)

    print(f"Found {len(student_courses)} courses for student {reg_no}")

    # 3. Compute skill profile
    profile = compute_skill_profile(student_courses, mapping)

    # 4. Print results sorted by score
    print("\n=== Skill Profile (Main Skills) ===")
    for skill, info in sorted(profile.items(), key=lambda x: x[1]["score"], reverse=True):
        print(f"{skill:45s}  Score: {info['score']:.2f}  (from {info['n_courses']} course(s))")
