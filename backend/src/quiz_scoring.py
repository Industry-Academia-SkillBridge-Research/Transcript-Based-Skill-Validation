# src/quiz_scoring.py

import os
from typing import Tuple

import pandas as pd

QUESTIONS_PATH = "output/quiz_questions_generated.csv"
RESPONSES_PATH = "output/quiz_responses_sample.csv"
RESULTS_PATH = "output/quiz_results_scored.csv"
SKILL_UPDATES_PATH = "output/skill_quiz_updates.csv"


def normalize_text(s: str) -> str:
    """
    Normalize question text to make matching robust:
    - convert to string
    - strip leading/trailing whitespace
    - collapse multiple spaces/newlines into a single space
    - lowercase
    """
    if pd.isna(s):
        return ""
    s = str(s)

    import re

    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def load_questions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Only the truly essential columns are required
    required = {
        "StudentID",
        "QuestionText",
        "CorrectOption",
        "Skill",
        "RoleName",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Questions file is missing columns: {missing}")

    # If QuestionID is missing, create a simple one
    if "QuestionID" not in df.columns:
        df["QuestionID"] = range(1, len(df) + 1)

    # If TargetDifficulty is missing, set a default
    if "TargetDifficulty" not in df.columns:
        df["TargetDifficulty"] = "Unknown"

    # Build normalized key for robust matching with responses
    df["QuestionKey"] = df["QuestionText"].apply(normalize_text)
    return df


def load_responses(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"StudentID", "QuestionText", "SelectedOption", "ResponseTimeSeconds"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Responses file is missing columns: {missing}")

    df["QuestionKey"] = df["QuestionText"].apply(normalize_text)
    df["ResponseTimeSeconds"] = pd.to_numeric(df["ResponseTimeSeconds"], errors="coerce")

    return df


def score_responses(
    questions_df: pd.DataFrame, responses_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Join responses to questions, compute correctness and per-skill quiz updates.

    Returns:
      - detailed_results_df: one row per answered question
      - skill_updates_df: one row per (StudentID, Skill) with aggregated quiz signal
    """
    # Inner join on StudentID + normalized QuestionKey
    merged = responses_df.merge(
        questions_df,
        on=["StudentID", "QuestionKey"],
        how="inner",
        suffixes=("_resp", "_q"),
    )

    if merged.empty:
        print("No overlapping rows between responses and questions AFTER normalization.")
        return pd.DataFrame(), pd.DataFrame()

    print(f"Merged response-question rows: {len(merged)}")

    # Compute correctness
    merged["IsCorrect"] = merged["SelectedOption"].astype(str).str.upper() == merged[
        "CorrectOption"
    ].astype(str).str.upper()

    merged["QuestionScore"] = merged["IsCorrect"].astype(float)

    # Optional: time-based penalty (you can tune this later)
    max_reasonable_time = 120.0
    merged["TimePenalty"] = (
        merged["ResponseTimeSeconds"].clip(upper=max_reasonable_time) / max_reasonable_time
    )
    merged["EffectiveScore"] = merged["QuestionScore"] - 0.2 * merged["TimePenalty"]

    # Rename QuestionText_q to QuestionText for clarity
    merged = merged.rename(columns={"QuestionText_q": "QuestionText"})

    detailed_cols = [
        "StudentID",
        "RoleName",
        "Skill",
        "QuestionID",
        "QuestionText",
        "SelectedOption",
        "CorrectOption",
        "ResponseTimeSeconds",
        "IsCorrect",
        "QuestionScore",
        "EffectiveScore",
        "TargetDifficulty",
    ]
    detailed_results = merged[detailed_cols].copy()

    # Aggregate per (StudentID, Skill)
    grouped = merged.groupby(["StudentID", "Skill"], as_index=False).agg(
        NumQuestions=("QuestionID", "count"),
        NumCorrect=("IsCorrect", "sum"),
        AvgTime=("ResponseTimeSeconds", "mean"),
        AvgQuestionScore=("QuestionScore", "mean"),
        AvgEffectiveScore=("EffectiveScore", "mean"),
    )

    grouped["QuizProficiency"] = grouped["AvgQuestionScore"].clip(0, 1)

    skill_updates = grouped.copy()
    return detailed_results, skill_updates


def main():
    print(f"Loading questions from: {QUESTIONS_PATH}")
    questions_df = load_questions(QUESTIONS_PATH)
    print(f"Questions: {len(questions_df)}")

    print(f"Loading responses from: {RESPONSES_PATH}")
    responses_df = load_responses(RESPONSES_PATH)
    print(f"Responses: {len(responses_df)}")

    detailed_df, updates_df = score_responses(questions_df, responses_df)

    os.makedirs("output", exist_ok=True)

    detailed_df.to_csv(RESULTS_PATH, index=False)
    updates_df.to_csv(SKILL_UPDATES_PATH, index=False)

    print(f"\nSaved detailed quiz results to: {RESULTS_PATH}")
    print(f"Rows: {len(detailed_df)}")

    print(f"\nSaved skill updates from quiz to: {SKILL_UPDATES_PATH}")
    print(f"Rows: {len(updates_df)}")

    if not detailed_df.empty:
        print("\nSample detailed results:")
        print(detailed_df.head())

    if not updates_df.empty:
        print("\nSample skill updates:")
        print(updates_df.head())


if __name__ == "__main__":
    main()
