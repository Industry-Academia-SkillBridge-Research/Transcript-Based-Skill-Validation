"""
run_full_pipeline_for_student.py

Simple orchestrator for the backend pipeline.

Usage examples:

  # Phase 1: transcript -> skills -> readiness -> quiz questions
  python run_full_pipeline_for_student.py --student-id IT21001288 --phase pre_quiz

  # Phase 2: after quiz responses are collected
  python run_full_pipeline_for_student.py --student-id IT21001288 --phase post_quiz
"""

import argparse
import os
from pathlib import Path

# All imports assume this file lives inside src/
import transcript_loader
import skill_aggregation_explainable
import job_postings_ingestion
import job_role_model_dynamic
import quiz_planner
import quiz_generation_rag
import quiz_scoring
import skill_profile_fusion


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"


def run_pre_quiz_pipeline(student_id: str) -> None:
    """
    Run the pipeline up to quiz generation.
    This stage does NOT require any quiz responses yet.
    """

    print("=" * 80)
    print("STEP 1: Transcript -> long format")
    print("=" * 80)
    transcript_loader.main()

    print("=" * 80)
    print("STEP 2: Long format -> skill profiles (explainable)")
    print("=" * 80)
    skill_aggregation_explainable.main()

    print("=" * 80)
    print("STEP 3: Job postings -> dynamic role skill templates")
    print("=" * 80)
    job_postings_ingestion.main()

    print("=" * 80)
    print("STEP 4: Compute role readiness (transcript driven)")
    print("=" * 80)
    job_role_model_dynamic.main()

    print("=" * 80)
    print("STEP 5: Build quiz plans from weak skills")
    print("=" * 80)
    quiz_planner.main()

    print("=" * 80)
    print("STEP 6: Generate quiz questions with RAG pipeline")
    print("=" * 80)
    quiz_generation_rag.main()

    print("\nPre-quiz pipeline finished.")
    print(f"Questions are in: {OUTPUT_DIR / 'quiz_questions_generated.csv'}")
    print(f"You can now deliver these questions to student {student_id} and")
    print("save their responses into output/quiz_responses.csv")


def run_post_quiz_pipeline(student_id: str) -> None:
    """
    Run the pipeline from quiz responses to fused profile and updated readiness.
    Expects that output/quiz_responses.csv already exists.
    """

    responses_path = OUTPUT_DIR / "quiz_responses.csv"
    if not responses_path.exists():
        print(f"Expected quiz responses at {responses_path}, but file not found.")
        print("Please export the student's responses from the front end first.")
        return

    print("=" * 80)
    print("STEP 7: Score quiz responses and derive skill updates")
    print("=" * 80)
    quiz_scoring.main()

    print("=" * 80)
    print("STEP 8: Fuse transcript skills with quiz-based proficiency")
    print("=" * 80)
    skill_profile_fusion.main()

    print("=" * 80)
    print("STEP 9: Recompute role readiness using fused skill profiles")
    print("=" * 80)
    job_role_model_dynamic.main()

    print("\nPost-quiz pipeline finished.")
    print(f"Updated fused skills: {OUTPUT_DIR / 'skill_profiles_with_quiz.csv'}")
    print(f"Updated readiness:    {OUTPUT_DIR / 'role_readiness_dynamic.csv'}")
    print(f"Check the rows for StudentID = {student_id} in these files.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end pipeline for transcript based skill validation."
    )
    parser.add_argument(
        "--student-id",
        type=str,
        required=True,
        help="StudentID (for example IT21001288). Used mainly for reporting.",
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["pre_quiz", "post_quiz"],
        required=True,
        help="Pipeline phase to run. "
             "pre_quiz: up to question generation. "
             "post_quiz: from responses to fused readiness.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    student_id = args.student_id.strip()

    print(f"Running pipeline for StudentID = {student_id}")
    print(f"Phase = {args.phase}")

    # Make sure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.phase == "pre_quiz":
        run_pre_quiz_pipeline(student_id)
    elif args.phase == "post_quiz":
        run_post_quiz_pipeline(student_id)


if __name__ == "__main__":
    main()
