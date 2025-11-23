from typing import Dict, List, Any
import numpy as np

from course_skill_mapping import load_course_skill_mapping
from transcript_loader import (
    load_transcripts,
    get_student_row,
    extract_student_courses,
)
from skill_profile import compute_skill_profile


# -------------------------------------------------------------------
# 1. Job role model (expert / config layer)
# -------------------------------------------------------------------

# Each role: {skill_name: importance_weight}
# Make sure these skill names match your MainSkill values
JOB_ROLE_MODEL: Dict[str, Dict[str, float]] = {
    "Data Analyst": {
        "Statistical Modeling & Data Analysis": 1.0,
        "Probability & Statistics for IT": 0.9,
        "Data Engineering & BI Analytics": 0.8,
        "Relational Database Design & SQL": 0.8,
        "Information Systems & Data Modelling": 0.7,
        "Programming Fundamentals & C Language": 0.6,
        "Machine Learning & Optimization": 0.6,
    },
    "Data Scientist": {
        "Machine Learning & Optimization": 1.0,
        "Statistical Modeling & Data Analysis": 0.95,
        "Probability & Statistics for IT": 0.9,
        "Information Retrieval & Web Analytics": 0.7,
        "Data Engineering & BI Analytics": 0.8,
        "Relational Database Design & SQL": 0.7,
        "Programming Fundamentals & C Language": 0.7,
    },
    "Software Engineer": {
        "Programming Fundamentals & C Language": 1.0,
        "Object-Oriented Programming Fundamentals": 0.9,
        "Software Engineering & Design": 0.9,
        "Data Structures & Algorithms": 0.95,
        "Web Development & Frameworks": 0.8,
        "Relational Database Design & SQL": 0.7,
        "Computer Systems & Networks": 0.6,
    },
    "Database Engineer": {
        "Relational Database Design & SQL": 1.0,
        "Database Systems & Storage": 0.95,
        "Data Engineering & BI Analytics": 0.8,
        "Information Systems & Data Modelling": 0.8,
        "Programming Fundamentals & C Language": 0.6,
    },
    "Business Intelligence Engineer": {
        "Data Engineering & BI Analytics": 1.0,
        "Relational Database Design & SQL": 0.9,
        "Statistical Modeling & Data Analysis": 0.8,
        "Database Systems & Storage": 0.7,
        "Information Systems & Data Modelling": 0.7,
    },
}


# -------------------------------------------------------------------
# 2. Helper functions
# -------------------------------------------------------------------

def flatten_skill_profile(profile: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Convert nested skill profile into a simple mapping: skill -> score.

    Input example:
        {
          "Programming Fundamentals & C Language": {"score": 0.78, "n_courses": 3},
          ...
        }

    Output:
        {
          "Programming Fundamentals & C Language": 0.78,
          ...
        }
    """
    return {skill: info["score"] for skill, info in profile.items()}


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors in [0, 1].

    Used to measure how close the student's skill vector is to the role's
    required skill vector.
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


# -------------------------------------------------------------------
# 3. Role readiness and gap analysis
# -------------------------------------------------------------------

def compute_role_readiness(
    student_skills: Dict[str, float],
    role_requirements: Dict[str, float],
    gap_threshold: float = 0.7,
) -> Dict[str, Any]:
    """
    Compare student's skill scores against a single role's required skills.

    Args:
        student_skills: {skill_name: score} from skill_profile
        role_requirements: {skill_name: importance_weight}
        gap_threshold: fraction of required weight below which a skill
                       is considered a gap (e.g., 0.7 means < 70% of required)

    Returns:
        {
          "readiness": float in [0, 1],
          "missing_skills": [
              {"skill": str, "student": float, "required": float},
              ...
          ]
        }
    """

    skills = list(role_requirements.keys())

    student_vec: List[float] = []
    role_vec: List[float] = []

    missing_details: List[Dict[str, float]] = []

    for skill in skills:
        required = role_requirements[skill]
        student_score = student_skills.get(skill, 0.0)

        student_vec.append(student_score)
        role_vec.append(required)

        # Record gaps for feedback
        if student_score < gap_threshold * required:
            missing_details.append(
                {
                    "skill": skill,
                    "student": student_score,
                    "required": required,
                }
            )

    student_vec_np = np.array(student_vec, dtype=float)
    role_vec_np = np.array(role_vec, dtype=float)

    readiness = cosine_similarity(student_vec_np, role_vec_np)

    # sort missing skills by importance (required weight)
    missing_details_sorted = sorted(
        missing_details, key=lambda x: x["required"], reverse=True
    )

    return {
        "readiness": readiness,
        "missing_skills": missing_details_sorted,
    }


def match_student_to_roles(
    student_profile: Dict[str, Dict[str, float]],
    job_role_model: Dict[str, Dict[str, float]] = JOB_ROLE_MODEL,
    top_k: int = 3,
    gap_threshold: float = 0.7,
) -> List[Dict[str, Any]]:
    """
    Compute readiness scores for all roles and return top_k best matches.

    Args:
        student_profile: nested dict from compute_skill_profile
        job_role_model: job role → {skill_name: importance_weight}
        top_k: number of top roles to return
        gap_threshold: threshold for skill gaps

    Returns:
        [
          {
            "role": "Data Analyst",
            "readiness": 0.81,
            "missing_skills": [
                {"skill": "...", "student": 0.45, "required": 0.9},
                ...
            ],
          },
          ...
        ]
    """
    flat_skills = flatten_skill_profile(student_profile)

    results: List[Dict[str, Any]] = []

    for role_name, requirements in job_role_model.items():
        role_result = compute_role_readiness(
            flat_skills, requirements, gap_threshold=gap_threshold
        )
        results.append(
            {
                "role": role_name,
                "readiness": role_result["readiness"],
                "missing_skills": role_result["missing_skills"],
            }
        )

    # sort roles by readiness descending
    results_sorted = sorted(results, key=lambda x: x["readiness"], reverse=True)
    return results_sorted[:top_k]


# -------------------------------------------------------------------
# 4. End-to-end test
# -------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Load static data
    course_mapping = load_course_skill_mapping("input/course_skill_mapping.csv")
    df = load_transcripts("input/transcript_data.csv")

    # 2. Pick a student to test
    #    Change this to any valid RegNo in your transcript_data.csv
    reg_no = "IT21709618"

    student_row = get_student_row(df, reg_no)
    student_courses = extract_student_courses(student_row)

    print(f"Found {len(student_courses)} courses for {reg_no}")

    # 3. Compute skill profile from courses
    skill_profile = compute_skill_profile(student_courses, course_mapping)

    # 4. Match profile to job roles
    matches = match_student_to_roles(skill_profile, JOB_ROLE_MODEL, top_k=3)

    print("\n=== Top Recommended Roles ===")
    for m in matches:
        print(f"\nRole: {m['role']}")
        print(f"Readiness: {m['readiness']:.2f}")

        print("Key skill gaps (top 3):")
        if not m["missing_skills"]:
            print("  (No major gaps for this role based on current threshold.)")
        else:
            for gap in m["missing_skills"][:3]:
                print(
                    f"  - {gap['skill']}: "
                    f"student={gap['student']:.2f}, required={gap['required']:.2f}"
                )
