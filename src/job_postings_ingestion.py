import math
import os
from typing import Dict, List

import pandas as pd

from course_skill_mapping import load_course_skill_mapping

def canonicalize_role(title: str) -> str:
    """
    Map noisy job titles to canonical role names.
    This is a simple rule-based normalizer.
    """
    t = str(title).lower()

    if "data scientist" in t or "data science" in t:
        return "Data Scientist"
    if "data analyst" in t or "business intelligence analyst" in t:
        return "Data Analyst"
    if "machine learning engineer" in t or "ml engineer" in t:
        return "ML Engineer"
    if "software engineer" in t or "developer" in t:
        return "Software Engineer"
    if "qa engineer" in t or "quality assurance" in t or "tester" in t:
        return "QA Engineer"

    # fallback: cleaned title (capitalized)
    return title.strip()

def build_skill_vocabulary(course_mapping: Dict[str, dict]) -> List[str]:
    """
    Build a list of canonical skills from course_skill_mapping.
    """
    skills = set()
    for course in course_mapping.values():
        for s in course["skills"]:
            skills.add(s.strip())
        # main skill is also included
        if course.get("main_skill"):
            skills.add(course["main_skill"].strip())
    return sorted(skills)


def text_contains_skill(text: str, skill: str) -> bool:
    """
    Very simple phrase-based matching.
    For now: check if all tokens of the skill phrase are in the text.
    You can later replace this with spaCy / embeddings.
    """
    text_l = text.lower()
    tokens = [tok for tok in skill.lower().split() if tok not in {"&", "and", "/", "-", ","}]

    return all(tok in text_l for tok in tokens) if tokens else False


def extract_skills_from_job(
    title: str,
    description: str,
    skill_vocab: List[str],
) -> List[str]:
    """
    Extract matching skills from a job posting using the skill vocabulary.
    """
    combined = f"{title} {description}".lower()

    matched = []
    for skill in skill_vocab:
        if text_contains_skill(combined, skill):
            matched.append(skill)

    return matched


def build_job_skill_table(
    postings_df: pd.DataFrame,
    skill_vocab: List[str],
) -> pd.DataFrame:
    """
    Build a table with one row per (JobID, RoleName, Skill).
    """
    records: List[dict] = []

    for _, row in postings_df.iterrows():
        job_id = row.get("JobID")
        title = row.get("Title", "")
        description = row.get("Description", "")

        if pd.isna(description):
            description = ""

        role_name = canonicalize_role(title)
        matched_skills = extract_skills_from_job(title, description, skill_vocab)

        for skill in matched_skills:
            records.append(
                {
                    "JobID": job_id,
                    "RoleName": role_name,
                    "Skill": skill,
                }
            )

    return pd.DataFrame(records)


def build_role_skill_templates(job_skill_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate job-skill matches into dynamic role skill templates.

    For each RoleName + Skill we compute:
      - JobCount: #jobs for that role where the skill appeared
      - RolePostingCount: total jobs for that role
      - Support: JobCount / RolePostingCount
      - Importance: Support * log(1 + RolePostingCount)
    """
    if job_skill_df.empty:
        return pd.DataFrame()

    # number of postings per role
    role_counts = (
        job_skill_df.groupby("RoleName")["JobID"].nunique().reset_index()
    )
    role_counts = role_counts.rename(columns={"JobID": "RolePostingCount"})

    # number of postings where each (RoleName, Skill) appears
    role_skill_counts = (
        job_skill_df.groupby(["RoleName", "Skill"])["JobID"]
        .nunique()
        .reset_index()
    )
    role_skill_counts = role_skill_counts.rename(columns={"JobID": "JobCount"})

    # merge to get RolePostingCount for each RoleName
    templates = role_skill_counts.merge(role_counts, on="RoleName", how="left")

    # compute support & importance
    templates["Support"] = (
        templates["JobCount"] / templates["RolePostingCount"].clip(lower=1)
    )

    # Importance gives more weight to skills that appear in many postings
    templates["Importance"] = templates["Support"] * templates[
        "RolePostingCount"
    ].apply(lambda n: math.log(1 + n))

    # normalize importance to [0, 1] per role for easier interpretation
    def normalize_group(group: pd.DataFrame) -> pd.DataFrame:
        max_imp = group["Importance"].max()
        if max_imp > 0:
            group["ImportanceNorm"] = group["Importance"] / max_imp
        else:
            group["ImportanceNorm"] = 0.0
        return group

    templates = templates.groupby("RoleName", group_keys=False).apply(normalize_group)

    return templates


def main():
    # 1) load job postings
    postings_df = pd.read_csv("input/job_postings_sample.csv")
    print(f"Loaded {len(postings_df)} job postings")

    # 2) load course → skill mapping and build skill vocabulary
    course_mapping = load_course_skill_mapping("input/course_skill_mapping.csv")
    skill_vocab = build_skill_vocabulary(course_mapping)
    print(f"Skill vocabulary size: {len(skill_vocab)}")

    # 3) build (JobID, RoleName, Skill) table
    job_skill_df = build_job_skill_table(postings_df, skill_vocab)
    print(f"Job-skill rows: {len(job_skill_df)}")

    # 4) build role-level templates
    templates_df = build_role_skill_templates(job_skill_df)
    print(f"Role-skill template rows: {len(templates_df)}")

    # 5) save
    os.makedirs("output", exist_ok=True)
    job_skill_df.to_csv("output/job_skill_matches.csv", index=False)
    templates_df.to_csv("output/job_role_skill_templates_dynamic.csv", index=False)

    print("\nSample role templates:")
    print(templates_df.head())


if __name__ == "__main__":
    main()