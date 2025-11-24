from typing import Dict, Any, List

import json

from course_skill_mapping import load_course_skill_mapping
from transcript_loader import (
    load_transcripts,
    get_student_row,
    extract_student_courses,
)
from skill_profile import compute_skill_profile
from job_role_matching import match_student_to_roles, JOB_ROLE_MODEL


# ----------------------------------------------------
# Helper: convert numeric score → skill level label
# ----------------------------------------------------

def skill_score_to_level(score: float) -> str:
    """
    Map continuous skill score to a human-readable level.

    Thresholds can be tuned based on empirical data later.
    """
    if score >= 0.85:
        return "Advanced"
    if score >= 0.70:
        return "Proficient"
    if score >= 0.50:
        return "Developing"
    if score > 0.0:
        return "Beginner"
    return "No Evidence"


def readiness_to_label(readiness: float) -> str:
    """
    Label overall readiness for a job role from cosine similarity score.
    """
    if readiness >= 0.85:
        return "Highly Ready"
    if readiness >= 0.70:
        return "Moderately Ready"
    if readiness > 0.50:
        return "Partially Ready"
    return "Not Ready"


# ----------------------------------------------------
# Core: build portfolio for one student
# ----------------------------------------------------

def build_student_portfolio(
    reg_no: str,
    transcripts_path: str = "input/transcript_data.csv",
    mapping_path: str = "input/course_skill_mapping.csv",
    job_role_model: Dict[str, Dict[str, float]] = JOB_ROLE_MODEL,
) -> Dict[str, Any]:
    """
    Full pipeline for one student:
      transcripts -> courses -> skills -> job roles -> portfolio dict
    """
    # 1. Load data
    course_mapping = load_course_skill_mapping(mapping_path)
    df = load_transcripts(transcripts_path)

    # 2. Get this student's transcript row
    student_row = get_student_row(df, reg_no)
    name = str(student_row.get("Name", "")).strip()

    # 3. Extract all their course records
    student_courses = extract_student_courses(student_row)

    # 4. Compute skill profile
    raw_skill_profile = compute_skill_profile(student_courses, course_mapping)
    # raw_skill_profile: {skill: {"score": float, "n_courses": int}}

    # 5. Build skill list with level labels
    skills_block: List[Dict[str, Any]] = []
    for skill_name, info in raw_skill_profile.items():
        score = float(info["score"])
        n_courses = int(info["n_courses"])
        level = skill_score_to_level(score)

        skills_block.append(
            {
                "name": skill_name,
                "score": score,
                "level": level,
                "n_courses": n_courses,
            }
        )

    # sort skills by score descending
    skills_block.sort(key=lambda x: x["score"], reverse=True)

    # 6. Match to job roles
    role_matches = match_student_to_roles(raw_skill_profile, job_role_model, top_k=3)

    roles_block: List[Dict[str, Any]] = []
    for match in role_matches:
        readiness = float(match["readiness"])
        label = readiness_to_label(readiness)

        # take only top 3 gaps to keep output readable
        top_gaps = match["missing_skills"][:3]

        roles_block.append(
            {
                "role": match["role"],
                "readiness": readiness,
                "label": label,
                "top_gaps": top_gaps,
            }
        )

    # 7. Build final portfolio object
    portfolio: Dict[str, Any] = {
        "reg_no": reg_no,
        "name": name,
        "skills": skills_block,
        "recommended_roles": roles_block,
    }

    return portfolio


# ----------------------------------------------------
# Simple CLI test
# ----------------------------------------------------

if __name__ == "__main__":
    # choose a RegNo that exists in your transcript_data.csv
    reg_no = "IT21709618"

    portfolio = build_student_portfolio(reg_no)

    print(f"=== Portfolio for {portfolio['name']} ({portfolio['reg_no']}) ===\n")

    print("Top skills:")
    for s in portfolio["skills"][:8]:
        print(
            f"  - {s['name']}: score={s['score']:.2f}, "
            f"level={s['level']}, courses={s['n_courses']}"
        )

    print("\nRecommended roles:")
    for r in portfolio["recommended_roles"]:
        print(
            f"  * {r['role']} -> readiness={r['readiness']:.2f} "
            f"({r['label']})"
        )
        if r["top_gaps"]:
            print("    Key gaps:")
            for g in r["top_gaps"]:
                print(
                    f"      - {g['skill']}: "
                    f"student={g['student']:.2f}, required={g['required']:.2f}"
                )
        else:
            print("    No major gaps at current threshold.")

    # optionally dump to JSON file for frontend or later analysis
    output_path = f"output/portfolio_{reg_no}.json"
    import os

    os.makedirs("output", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(portfolio, f, indent=2)
    print(f"\n[INFO] Saved portfolio JSON to {output_path}")
