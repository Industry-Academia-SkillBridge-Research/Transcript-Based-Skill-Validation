import os
from typing import Tuple

import pandas as pd


SKILL_PROFILE_PATH = "output/skill_profiles_explainable.csv"
QUIZ_UPDATES_PATH = "output/skill_quiz_updates.csv"
OUTPUT_PATH = "output/skill_profiles_with_quiz.csv"


def load_skill_profiles(path: str) -> pd.DataFrame:
    """
    Load baseline transcript-derived skill profile.

    Expected columns (from skill_aggregation_explainable.py):
      StudentID, Skill, EvidenceCount, TotalContribution,
      ScoreNormalized, SkillLevel
    """
    df = pd.read_csv(path)
    required = {"StudentID", "Skill", "ScoreNormalized"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Skill profile file missing columns: {missing}")
    return df


def load_quiz_updates(path: str) -> pd.DataFrame:
    """
    Load aggregated quiz proficiency per (StudentID, Skill).

    Expected columns (from quiz_scoring.py):
      StudentID, Skill, NumQuestions, NumCorrect,
      NumIncorrect, AvgEffectiveScore, QuizProficiency
    """
    if not os.path.exists(path):
        print(f"[WARN] Quiz updates file not found: {path}")
        return pd.DataFrame(columns=[
            "StudentID", "Skill", "NumQuestions",
            "NumCorrect", "NumIncorrect",
            "AvgEffectiveScore", "QuizProficiency",
        ])

    df = pd.read_csv(path)
    if df.empty:
        print(f"[WARN] Quiz updates file is empty: {path}")
        return df

    required = {"StudentID", "Skill", "QuizProficiency"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Quiz updates file missing columns: {missing}")

    return df


def map_score_to_level(score: float) -> str:
    """
    Map final skill score in [0, 1] to a qualitative level.
    This will overwrite the old SkillLevel with a quiz-aware one.

    Thresholds are your design choice. These are reasonable:
      0.00–0.25 -> Beginner
      0.25–0.50 -> Developing
      0.50–0.75 -> Proficient
      0.75–1.00 -> Advanced
    """
    if score < 0.25:
        return "Beginner"
    elif score < 0.50:
        return "Developing"
    elif score < 0.75:
        return "Proficient"
    else:
        return "Advanced"


def fuse_profiles(
    base_df: pd.DataFrame,
    quiz_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine transcript-based skill scores with quiz-based proficiency.

    Idea:
      - Transcript ScoreNormalized = prior belief (static evidence from grades)
      - QuizProficiency = new evidence (dynamic, targeted assessment)

    We increase the weight of quiz evidence as the number of questions grows:
      quiz_weight = min(0.7, NumQuestions / 10.0)
      transcript_weight = 1 - quiz_weight

    So:
      - 1 question -> quiz has small influence
      - many questions (>= 7–10) -> quiz dominates up to 70%
    """
    if quiz_df.empty:
        print("No quiz updates available. Returning base skill profile unchanged.")
        fused = base_df.copy()
        fused["FinalScore"] = fused["ScoreNormalized"]
        fused["FinalSkillLevel"] = fused["ScoreNormalized"].apply(map_score_to_level)
        fused["TranscriptWeight"] = 1.0
        fused["QuizWeight"] = 0.0
        fused["QuizProficiency"] = None
        fused["NumQuestions"] = 0
        return fused

    # Merge on StudentID + Skill (left join: keep all baseline skills)
    merged = base_df.merge(
        quiz_df,
        on=["StudentID", "Skill"],
        how="left",
        suffixes=("", "_quiz"),
    )

    # Fill NaN for quiz metrics where no quiz was taken
    merged["NumQuestions"] = merged.get("NumQuestions", 0).fillna(0).astype(int)
    merged["QuizProficiency"] = merged["QuizProficiency"].fillna(pd.NA)

    # Compute dynamic weights
    # quiz_weight in [0, 0.7], increases with NumQuestions
    merged["QuizWeight"] = (merged["NumQuestions"] / 10.0).clip(lower=0.0, upper=0.7)

    # If there is no quiz proficiency (NaN), weight must be zero
    merged.loc[merged["QuizProficiency"].isna(), "QuizWeight"] = 0.0

    merged["TranscriptWeight"] = 1.0 - merged["QuizWeight"]

    # For rows without quiz, set QuizProficiency = 0 (won't matter due to weight=0)
    merged["QuizProficiencyFilled"] = merged["QuizProficiency"].fillna(0.0)

    # Final fused score
    merged["FinalScore"] = (
        merged["TranscriptWeight"] * merged["ScoreNormalized"]
        + merged["QuizWeight"] * merged["QuizProficiencyFilled"]
    )

    # Clip to [0,1] to be safe
    merged["FinalScore"] = merged["FinalScore"].clip(lower=0.0, upper=1.0)

    # New skill level based on FinalScore
    merged["FinalSkillLevel"] = merged["FinalScore"].apply(map_score_to_level)

    return merged


def main():
    print(f"Loading baseline skill profiles from: {SKILL_PROFILE_PATH}")
    base_df = load_skill_profiles(SKILL_PROFILE_PATH)
    print(f"Baseline rows: {len(base_df)}")

    print(f"Loading quiz updates from: {QUIZ_UPDATES_PATH}")
    quiz_df = load_quiz_updates(QUIZ_UPDATES_PATH)
    print(f"Quiz update rows: {len(quiz_df)}")

    fused_df = fuse_profiles(base_df, quiz_df)
    print(f"Fused rows: {len(fused_df)}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fused_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved fused skill profiles to: {OUTPUT_PATH}")

    # Show a sample for the student you tested (if available)
    example_id = "IT21001288"
    subset = fused_df[fused_df["StudentID"] == example_id]
    if not subset.empty:
        print(f"\nFused skills for {example_id}:")
        print(
            subset[
                [
                    "Skill",
                    "ScoreNormalized",
                    "QuizProficiency",
                    "NumQuestions",
                    "TranscriptWeight",
                    "QuizWeight",
                    "FinalScore",
                    "FinalSkillLevel",
                ]
            ]
            .sort_values("FinalScore", ascending=False)
            .head(10)
        )


if __name__ == "__main__":
    main()
