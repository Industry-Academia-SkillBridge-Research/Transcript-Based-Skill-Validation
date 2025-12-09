"""
quiz_generation_rag.py

Purpose:
    Take quiz plans (which skills to assess, how many questions, difficulty)
    + a skill knowledge corpus,
    then use retrieval (TF-IDF) to build context for each skill and
    generate quiz questions.

    This is the RAG-style quiz generation pipeline for your component.

Inputs:
    - output/quiz_plans.csv
        Columns: StudentID, RoleName, Skill, StudentLevel,
                 RequiredImportance, StudentScore, AttainedFraction,
                 TargetDifficulty, NumQuestions

    - content/skill_corpus.csv
        Columns: Skill, SourceType, SourceName, Content

Output:
    - output/quiz_questions_generated.csv
        Columns: StudentID, RoleName, Skill, Difficulty,
                 QuestionText, OptionA, OptionB, OptionC, OptionD,
                 CorrectOption, Explanation
"""

import os
from typing import List, Dict

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Loading utilities
# -----------------------------

def load_quiz_plans(path: str) -> pd.DataFrame:
    """
    Load quiz plans produced by quiz_planner.py.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Quiz plans file not found: {path}")

    df = pd.read_csv(path)
    expected_cols = {
        "StudentID",
        "RoleName",
        "Skill",
        "StudentLevel",
        "RequiredImportance",
        "StudentScore",
        "AttainedFraction",
        "TargetDifficulty",
        "NumQuestions",
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"quiz_plans file is missing columns: {missing}")

    return df


def load_skill_corpus(path: str) -> pd.DataFrame:
    """
    Load the knowledge corpus used for retrieval.

    Expected columns:
        - Skill: skill name (e.g. 'Hypothesis Testing')
        - SourceType: e.g. 'ModuleOutline', 'Textbook', 'Website'
        - SourceName: e.g. 'IT2110 - Probability & Statistics'
        - Content: short paragraph explaining the concept
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Skill corpus file not found: {path}. "
            f"Create content/skill_corpus.csv with columns: "
            f"Skill,SourceType,SourceName,Content"
        )

    df = pd.read_csv(path)
    expected_cols = {"Skill", "SourceType", "SourceName", "Content"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"skill_corpus file is missing columns: {missing}")

    # Drop rows with empty content
    df = df.dropna(subset=["Content"])
    return df


# -----------------------------
# Retrieval (TF-IDF based)
# -----------------------------

def build_tfidf_index(corpus_df: pd.DataFrame):
    """
    Build a TF-IDF index over the Content field.
    Returns:
        vectorizer, tfidf_matrix
    """
    contents = corpus_df["Content"].astype(str).tolist()
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.95,
        min_df=1,
        ngram_range=(1, 2),
    )
    tfidf_matrix = vectorizer.fit_transform(contents)
    return vectorizer, tfidf_matrix


def retrieve_context_for_skill(
    skill: str,
    corpus_df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    tfidf_matrix,
    max_paragraphs: int = 5,
) -> str:
    """
    Given a skill name, retrieve the most relevant paragraphs from the corpus.

    Steps:
        1. Optionally filter corpus by Skill column.
        2. Use TF-IDF + cosine similarity between the skill text
           and each Content paragraph.
        3. Take top-k paragraphs and join as context.

    This is the "R" in RAG.
    """
    if corpus_df.empty:
        return ""

    # First, try to filter by exact or partial skill name
    mask = corpus_df["Skill"].fillna("").str.contains(skill, case=False, na=False)
    filtered = corpus_df[mask]
    if filtered.empty:
        # Fallback: use entire corpus
        filtered = corpus_df.copy()
        indices = filtered.index.to_list()
    else:
        indices = filtered.index.to_list()

    if not indices:
        return ""

    # Build a small tfidf matrix corresponding to filtered rows
    # using the global tfidf_matrix
    sub_matrix = tfidf_matrix[indices, :]

    # Represent the skill text as a query vector
    skill_query = vectorizer.transform([skill])

    # Compute cosine similarity between skill and each paragraph
    sims = cosine_similarity(skill_query, sub_matrix).flatten()

    # Rank indices by similarity
    ranked = sorted(
        zip(indices, sims),
        key=lambda x: x[1],
        reverse=True,
    )

    top_indices = [idx for idx, _ in ranked[:max_paragraphs]]

    # Join selected paragraphs into a single context string
    paragraphs: List[str] = corpus_df.loc[top_indices, "Content"].astype(str).tolist()

    context = "\n\n".join(paragraphs)
    return context


# -----------------------------
# Question generation (placeholder)
# -----------------------------

def call_llm_generate_mcqs(prompt: str, num_questions: int) -> List[Dict]:
    """
    Placeholder for LLM-based MCQ generation.

    In the real system you would:
        - send `prompt` to an LLM (OpenAI API, local model, etc.)
        - ask it to respond in a strict JSON format
        - parse that JSON and return list of questions

    For now this function returns simple dummy questions so that
    the pipeline runs end-to-end without external APIs.
    """
    questions: List[Dict] = []

    for i in range(1, num_questions + 1):
        q_text = f"[DUMMY] Question {i}: This is a placeholder question generated for this skill."
        questions.append(
            {
                "QuestionText": q_text,
                "OptionA": "Option A",
                "OptionB": "Option B",
                "OptionC": "Option C",
                "OptionD": "Option D",
                "CorrectOption": "A",
                "Explanation": "This is a placeholder explanation. Replace this with LLM-generated content.",
            }
        )

    return questions


def generate_mcqs_for_plan_row(
    plan_row: pd.Series,
    corpus_df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    tfidf_matrix,
) -> List[Dict]:
    """
    Build a RAG prompt for one quiz plan row and call the LLM generator.

    Returns a list of question dicts with metadata attached.
    """
    skill = plan_row["Skill"]
    difficulty = plan_row["TargetDifficulty"]
    num_questions = int(plan_row["NumQuestions"])
    student_level = plan_row["StudentLevel"]

    # Retrieve knowledge context for the skill
    context = retrieve_context_for_skill(
        skill=skill,
        corpus_df=corpus_df,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        max_paragraphs=5,
    )

    if not context:
        context = f"No specific context found for skill: {skill}. Use general knowledge."

    # Build prompt for the LLM
    prompt = f"""
You are an exam question generator for undergraduate IT / Data Science students.

Skill to assess: {skill}
Student current level: {student_level}
Target difficulty: {difficulty}
Number of questions: {num_questions}

Use ONLY the following context snippets as your knowledge base.
Do not introduce concepts that are not supported by the context.

Context:
{context}

Task:
Generate {num_questions} {difficulty}-level multiple choice questions (MCQs) that test understanding and application of the skill.
For each question, provide:
1. The question text.
2. Four options: A, B, C, D.
3. The correct option letter.
4. A brief explanation based strictly on the context.

Respond in a structured JSON-like list, for example:
[
  {{
    "question": "...",
    "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "correct_option": "B",
    "explanation": "..."
  }},
  ...
]
"""

    # Call the LLM (currently a placeholder)
    raw_questions = call_llm_generate_mcqs(prompt, num_questions)

    # Attach metadata for each question
    enriched_questions: List[Dict] = []
    for q in raw_questions:
        record = {
            "StudentID": plan_row["StudentID"],
            "RoleName": plan_row["RoleName"],
            "Skill": skill,
            "Difficulty": difficulty,
        }
        record.update(q)
        enriched_questions.append(record)

    return enriched_questions


# -----------------------------
# Main pipeline
# -----------------------------

def main():
    quiz_plans_path = "output/quiz_plans.csv"
    skill_corpus_path = "content/skill_corpus.csv"
    output_path = "output/quiz_questions_generated.csv"

    print(f"Loading quiz plans from: {quiz_plans_path}")
    plans_df = load_quiz_plans(quiz_plans_path)
    print(f"Quiz plan rows: {len(plans_df)}")

    print(f"Loading skill corpus from: {skill_corpus_path}")
    corpus_df = load_skill_corpus(skill_corpus_path)
    print(f"Skill corpus rows: {len(corpus_df)}")

    print("Building TF-IDF index over skill corpus content...")
    vectorizer, tfidf_matrix = build_tfidf_index(corpus_df)

    # Optional: limit number of plan rows for testing
    MAX_PLANS = 30
    plans_to_process = plans_df.head(MAX_PLANS).copy()

    all_questions: List[Dict] = []

    for idx, row in plans_to_process.iterrows():
        student_id = row["StudentID"]
        skill = row["Skill"]
        role = row["RoleName"]
        print(f"Generating questions for Student {student_id}, Role '{role}', Skill '{skill}'")

        q_records = generate_mcqs_for_plan_row(
            plan_row=row,
            corpus_df=corpus_df,
            vectorizer=vectorizer,
            tfidf_matrix=tfidf_matrix,
        )
        all_questions.extend(q_records)

    if not all_questions:
        print("No questions generated. Check inputs and corpus.")
        return

    out_df = pd.DataFrame(all_questions)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)

    print(f"\nSaved {len(out_df)} questions to {output_path}")
    print("\nSample questions:")
    print(out_df.head(5))


if __name__ == "__main__":
    main()
