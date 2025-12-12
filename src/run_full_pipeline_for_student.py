import argparse
import os
import sys
import subprocess
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]


def run_step(name: str, script_rel_path: str, extra_args=None):
    """
    Helper to run another Python script using the *same* interpreter
    that is currently running this file (sys.executable).
    """
    script_path = BASE_DIR / script_rel_path
    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n=== Running step: {name} ===")
    print("Command:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"=== Finished: {name} ===\n")


def show_pre_quiz_summary(student_id: str):
    """
    After pre_quiz pipeline: show skill profile, top roles, and planned quiz skills.
    """
    skills_path = BASE_DIR / "output" / "skill_profiles_explainable.csv"
    roles_path = BASE_DIR / "output" / "role_readiness_dynamic.csv"
    quiz_plan_path = BASE_DIR / "output" / "quiz_plans.csv"

    if skills_path.exists():
        skills_df = pd.read_csv(skills_path)
        stu_skills = skills_df[skills_df["StudentID"] == student_id].copy()
        print(f"\n--- Skill profile (baseline) for {student_id} ---")
        print(
            stu_skills.sort_values("ScoreNormalized", ascending=False)
            .head(10)[["Skill", "ScoreNormalized", "SkillLevel"]]
        )
    else:
        print("\n[WARN] skill_profiles_explainable.csv not found")

    if roles_path.exists():
        roles_df = pd.read_csv(roles_path)
        stu_roles = roles_df[roles_df["StudentID"] == student_id].copy()
        print(f"\n--- Top roles for {student_id} (pre-quiz) ---")
        print(
            stu_roles.sort_values("ReadinessScore", ascending=False)
            .head(5)[
                [
                    "RoleName",
                    "ReadinessScore",
                    "Coverage",
                    "NumSkills",
                    "NumWeakOrMissing",
                ]
            ]
        )
    else:
        print("\n[WARN] role_readiness_dynamic.csv not found")

    if quiz_plan_path.exists():
        qp_df = pd.read_csv(quiz_plan_path)
        stu_qp = qp_df[qp_df["StudentID"] == student_id].copy()
        print(f"\n--- Quiz plan for {student_id} (first few skills) ---")
        print(
            stu_qp.head(10)[
                [
                    "RoleName",
                    "Skill",
                    "StudentLevel",
                    "TargetDifficulty",
                    "NumQuestions",
                ]
            ]
        )
    else:
        print("\n[WARN] quiz_plans.csv not found")


def show_post_quiz_summary(student_id: str):
    """
    After post_quiz pipeline: show quiz-based skill updates, fused profile,
    and updated top roles.
    """
    quiz_updates_path = BASE_DIR / "output" / "skill_quiz_updates.csv"
    fused_path = BASE_DIR / "output" / "skill_profiles_with_quiz.csv"
    roles_path = BASE_DIR / "output" / "role_readiness_dynamic.csv"

    if quiz_updates_path.exists():
        q_df = pd.read_csv(quiz_updates_path)
        stu_q = q_df[q_df["StudentID"] == student_id].copy()
        print(f"\n--- Quiz-based skill updates for {student_id} ---")
        if not stu_q.empty:
            print(
                stu_q.sort_values("FinalQuizScore", ascending=False)
                .head(10)[
                    [
                        "Skill",
                        "NumQuestions",
                        "AvgEffectiveScore",
                        "FinalQuizScore",
                        "QuizProficiency",
                    ]
                ]
            )
        else:
            print("No quiz updates for this student.")
    else:
        print("\n[WARN] skill_quiz_updates.csv not found")

    if fused_path.exists():
        fused_df = pd.read_csv(fused_path)
        stu_fused = fused_df[fused_df["StudentID"] == student_id].copy()
        print(f"\n--- Fused skill profile for {student_id} ---")
        print(
            stu_fused.sort_values("FinalScore", ascending=False)
            .head(10)[["Skill", "BaselineScore", "FinalScore", "FinalSkillLevel"]]
        )
    else:
        print("\n[WARN] skill_profiles_with_quiz.csv not found")

    if roles_path.exists():
        roles_df = pd.read_csv(roles_path)
        stu_roles = roles_df[roles_df["StudentID"] == student_id].copy()
        print(f"\n--- Updated top roles for {student_id} (post-quiz) ---")
        print(
            stu_roles.sort_values("ReadinessScore", ascending=False)
            .head(5)[
                [
                    "RoleName",
                    "ReadinessScore",
                    "Coverage",
                    "NumSkills",
                    "NumWeakOrMissing",
                ]
            ]
        )
    else:
        print("\n[WARN] role_readiness_dynamic.csv not found")


def main():
    parser = argparse.ArgumentParser(
        description="Run the full skill-validation pipeline for a single student."
    )
    parser.add_argument(
        "--student-id", required=True, help="StudentID / RegNo, e.g. IT21001288"
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=["pre_quiz", "post_quiz"],
        help="Which part of the pipeline to run.",
    )
    args = parser.parse_args()
    student_id = args.student_id
    phase = args.phase

    print(f"\n### Running pipeline for student: {student_id}, phase: {phase} ###")

    if phase == "pre_quiz":
        # 1. Transcript → long format
        run_step("Transcript loader", "src/transcript_loader.py")

        # 2. Course–skill mapping (sanity check; also useful for other scripts)
        run_step("Course–skill mapping check", "src/course_skill_mapping.py")

        # 3. Aggregate skills (explainable profile)
        run_step("Skill aggregation (explainable)", "src/skill_aggregation_explainable.py")

        # 4. Ingest job data (JSON → CSV → skill templates)
        run_step("Convert Job_data.json to CSV", "src/convert_job_json_to_csv.py")
        run_step("Job postings ingestion", "src/job_postings_ingestion.py")

        # 5. Fuse baseline + (any existing) quiz signals
        run_step("Skill profile fusion", "src/skill_profile_fusion.py")

        # 6. Compute role readiness using dynamic templates
        run_step("Role readiness (dynamic)", "src/job_role_model_dynamic.py")

        # 7. Build quiz plans for weak/missing skills
        run_step("Quiz planning", "src/quiz_planner.py")

        # 8. Generate quiz questions using RAG over skill corpus
        run_step("Quiz generation (RAG)", "src/quiz_generation_rag.py")

        # Show summary for this student
        show_pre_quiz_summary(student_id)

    elif phase == "post_quiz":
        # Here we assume:
        #  - Student has answered quizzes
        #  - Responses are stored in output/quiz_responses_*.csv (currently we use sample)
        run_step("Quiz scoring", "src/quiz_scoring.py")

        # Fuse transcript-based skills with quiz-based updates
        run_step("Skill profile fusion", "src/skill_profile_fusion.py")

        # Recompute role readiness after updated skills
        run_step("Role readiness (dynamic)", "src/job_role_model_dynamic.py")

        # Optionally generate a new round of quizzes (refinement cycle)
        run_step("Quiz planning (post-quiz)", "src/quiz_planner.py")
        run_step("Quiz generation (RAG, post-quiz)", "src/quiz_generation_rag.py")

        # Show summary for this student
        show_post_quiz_summary(student_id)

    print("\n### Pipeline run completed. ###")


if __name__ == "__main__":
    main()
