# Transcript-Based Skill Validation & Job Alignment

This project builds a backend pipeline that:

1. Reads university transcripts and converts course grades into a **skill profile** for each student.
2. Learns **job role skill templates** from real job postings.
3. Computes **role readiness scores** and highlights missing skills.
4. Generates **targeted quizzes** for weak skills using a RAG-style component over a curated skill corpus.
5. Scores quiz responses, fuses the result back into the skill profile, and **recomputes readiness**.

The focus is on transparent, explainable skill scoring that can be justified to academic staff.

---

## Project structure

Typical layout:

```text
Transcript-Based-Skill-Validation/
  src/
    transcript_loader.py
    course_skill_mapping.py
    skill_aggregation_explainable.py
    job_postings_ingestion.py
    job_role_model_dynamic.py
    quiz_planner.py
    generate_skill_corpus_template.py
    auto_fill_skill_corpus.py
    quiz_generation_rag.py
    quiz_scoring.py
    skill_profile_fusion.py
    run_full_pipeline_for_student.py   ← orchestration script

  input/
    transcript_data.csv
    course_skill_mapping.csv
    job_postings_sample.csv

  content/
    skill_corpus.csv                   ← curated skill texts for quiz generation

  output/
    transcripts_long.csv
    skill_profiles_explainable.csv
    job_postings_with_skills.csv
    job_role_skill_templates_dynamic.csv
    role_readiness_dynamic.csv
    role_readiness_explainable.csv
    quiz_plans.csv
    quiz_questions_generated.csv
    quiz_responses.csv
    quiz_results_scored.csv
    skill_quiz_updates.csv
    skill_profiles_with_quiz.csv

  requirements.txt
  README.md
