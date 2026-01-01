"""
Question Diversity Module

Ensures generated questions are diverse by:
1. Checking for near-duplicates using TF-IDF cosine similarity
2. Requiring mix of question types (definition, scenario, apply, compare)
"""

from typing import List, Dict, Tuple, Optional
import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def check_duplicate_questions(
    questions: List[Dict],
    similarity_threshold: float = 0.85
) -> Tuple[bool, List[str], List[Dict]]:
    """
    Check for near-duplicate questions using TF-IDF cosine similarity.
    
    Args:
        questions: List of question dictionaries with 'question' field
        similarity_threshold: Cosine similarity threshold (0-1). Questions above this are considered duplicates.
                              Default 0.85 means 85% similarity or higher is a duplicate.
    
    Returns:
        Tuple of:
        - is_diverse: True if no duplicates found, False otherwise
        - duplicate_warnings: List of warning messages
        - unique_questions: List of unique questions (duplicates removed)
    """
    if not SKLEARN_AVAILABLE:
        return True, ["sklearn not available - skipping duplicate check"], questions
    
    if len(questions) < 2:
        return True, [], questions
    
    # Extract question texts
    question_texts = []
    for q in questions:
        q_text = q.get("question", "") or q.get("QuestionText", "")
        if isinstance(q_text, str):
            question_texts.append(q_text.strip())
        else:
            question_texts.append("")
    
    # Remove empty questions
    valid_indices = [i for i, text in enumerate(question_texts) if text]
    if len(valid_indices) < 2:
        return True, [], questions
    
    valid_texts = [question_texts[i] for i in valid_indices]
    
    # Build TF-IDF vectors
    try:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),  # Unigrams and bigrams
            max_features=500
        )
        tfidf_matrix = vectorizer.fit_transform(valid_texts)
        
        # Compute pairwise cosine similarities
        similarities = cosine_similarity(tfidf_matrix)
        
        # Find duplicates (excluding self-similarity = 1.0)
        duplicates_found = []
        duplicate_indices = set()
        warnings = []
        
        for i in range(len(valid_indices)):
            for j in range(i + 1, len(valid_indices)):
                similarity = similarities[i][j]
                if similarity >= similarity_threshold:
                    orig_i = valid_indices[i]
                    orig_j = valid_indices[j]
                    duplicate_indices.add(orig_j)  # Keep first, remove second
                    
                    q1_text = valid_texts[i][:50] + "..." if len(valid_texts[i]) > 50 else valid_texts[i]
                    q2_text = valid_texts[j][:50] + "..." if len(valid_texts[j]) > 50 else valid_texts[j]
                    
                    warnings.append(
                        f"Duplicate detected (similarity: {similarity:.2f}): "
                        f"Q{i+1} '{q1_text}' and Q{j+1} '{q2_text}'"
                    )
                    duplicates_found.append((orig_i, orig_j, similarity))
        
        # Remove duplicates (keep first occurrence)
        unique_questions = [
            q for idx, q in enumerate(questions)
            if idx not in duplicate_indices
        ]
        
        is_diverse = len(duplicates_found) == 0
        
        if duplicates_found:
            warnings.insert(0, f"Found {len(duplicates_found)} duplicate question pair(s)")
        
        return is_diverse, warnings, unique_questions
        
    except Exception as e:
        # If TF-IDF fails, return original questions with warning
        return True, [f"Duplicate check failed: {str(e)}"], questions


def get_diversity_prompt_instruction() -> str:
    """
    Returns the prompt instruction for question diversity.
    
    This should be added to the Gemini prompt to encourage diverse question types.
    """
    return """
QUESTION DIVERSITY REQUIREMENT:
- Do NOT repeat question types. Mix different question formats:
  * Definition questions: "What is X?" or "Define Y"
  * Scenario-based questions: "In a situation where..." or "Given that..."
  * Application questions: "How would you apply X to..." or "Which approach..."
  * Comparison questions: "What is the difference between X and Y?" or "Which is better..."
- Ensure questions cover different aspects of the skill, not just variations of the same concept.
- Each question should test a distinct understanding or application of the skill.
""".strip()


def filter_duplicates_from_questions(
    questions: List[Dict],
    similarity_threshold: float = 0.85
) -> Tuple[List[Dict], List[str]]:
    """
    Filter out duplicate questions from a list.
    
    Args:
        questions: List of question dictionaries
        similarity_threshold: Similarity threshold for duplicate detection (default 0.85)
    
    Returns:
        Tuple of (filtered_questions, warnings)
    """
    is_diverse, warnings, unique_questions = check_duplicate_questions(
        questions,
        similarity_threshold=similarity_threshold
    )
    return unique_questions, warnings

