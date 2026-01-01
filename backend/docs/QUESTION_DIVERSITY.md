# Question Diversity Module

## Overview

Prevents Gemini from generating repetitive or near-duplicate questions by:

1. **Prompt Engineering**: Instructs Gemini to mix question types (definition, scenario, apply, compare)
2. **Duplicate Detection**: Uses TF-IDF cosine similarity to detect near-duplicate questions
3. **Automatic Filtering**: Removes duplicates automatically during generation

## Features

### 1. Question Type Diversity

The prompt instructs Gemini to mix different question formats:
- **Definition questions**: "What is X?" or "Define Y"
- **Scenario-based questions**: "In a situation where..." or "Given that..."
- **Application questions**: "How would you apply X to..." or "Which approach..."
- **Comparison questions**: "What is the difference between X and Y?" or "Which is better..."

### 2. Duplicate Detection

Uses **TF-IDF cosine similarity** to detect near-duplicate questions:

- **Algorithm**: TF-IDF vectorization with unigrams and bigrams
- **Similarity Threshold**: 0.85 (85% similarity = duplicate)
- **Processing**: Lowercase, removes stop words, handles similarity scoring

**Example:**
```
Question 1: "What is SQL?"
Question 2: "What does SQL mean?"
→ Similarity: 0.92 → DUPLICATE (removed)
```

### 3. Automatic Filtering

- Duplicates are automatically removed
- First occurrence is kept, subsequent duplicates are removed
- Warnings are logged for monitoring

## Implementation

### Module: `backend/src/question_diversity.py`

**Key Functions:**

1. **`check_duplicate_questions(questions, similarity_threshold=0.85)`**
   - Checks for duplicates in a list of questions
   - Returns: `(is_diverse, warnings, unique_questions)`

2. **`filter_duplicates_from_questions(questions, similarity_threshold=0.85)`**
   - Filters out duplicates
   - Returns: `(filtered_questions, warnings)`

3. **`get_diversity_prompt_instruction()`**
   - Returns prompt text to add to Gemini prompt
   - Encourages diverse question types

### Integration Points

1. **Quiz Generation (`quiz_generation_gemini.py`)**:
   - Adds diversity instruction to prompt
   - Checks for duplicates after validation
   - Removes duplicates automatically

2. **API Endpoint (`api/main.py`)**:
   - Final duplicate check across all questions (all skills)
   - Ensures no duplicates even across different skills

## Usage Example

```python
from src.question_diversity import check_duplicate_questions

questions = [
    {"question": "What is SQL?"},
    {"question": "What does SQL mean?"},
    {"question": "How do you write a SELECT query?"}
]

is_diverse, warnings, unique_questions = check_duplicate_questions(
    questions,
    similarity_threshold=0.85
)

# is_diverse: False
# warnings: ["Duplicate detected (similarity: 0.92): Q1 'What is SQL?' and Q2 'What does SQL mean?'"]
# unique_questions: [{"question": "What is SQL?"}, {"question": "How do you write a SELECT query?"}]
```

## Configuration

### Similarity Threshold

Default: `0.85` (85% similarity)

- **Lower (e.g., 0.70)**: More strict, removes more questions
- **Higher (e.g., 0.90)**: Less strict, only removes very similar questions

Adjust in:
- `check_duplicate_questions(..., similarity_threshold=0.85)`
- `filter_duplicates_from_questions(..., similarity_threshold=0.85)`

### TF-IDF Parameters

Current settings in `question_diversity.py`:
```python
TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    ngram_range=(1, 2),  # Unigrams and bigrams
    max_features=500
)
```

## How It Works

### Step 1: Prompt Engineering

Prompt includes diversity instruction:
```
QUESTION DIVERSITY REQUIREMENT:
- Do NOT repeat question types. Mix different question formats:
  * Definition questions: "What is X?"
  * Scenario-based questions: "In a situation where..."
  * Application questions: "How would you apply X to..."
  * Comparison questions: "What is the difference between X and Y?"
```

### Step 2: Question Generation

Gemini generates questions following diversity guidelines.

### Step 3: Duplicate Detection

1. Extract question texts
2. Build TF-IDF vectors (lowercase, stop words removed, ngrams)
3. Compute pairwise cosine similarities
4. Flag questions with similarity ≥ 0.85
5. Remove duplicates (keep first occurrence)

### Step 4: Result

- Unique questions returned
- Warnings logged (duplicate information)
- Quiz proceeds with diverse questions

## Performance

- **Time Complexity**: O(n²) for n questions (pairwise similarity)
- **Typical Runtime**: < 1 second for 10-20 questions
- **Memory**: Minimal (TF-IDF matrices are small)

## Limitations

1. **Semantic Similarity**: TF-IDF captures lexical similarity, not deep semantic similarity
   - Example: "What is SQL?" vs "Define SQL" → Detected ✅
   - Example: "How do you query data?" vs "How do you retrieve data?" → May not detect ❌

2. **Context Loss**: Only compares question text, not options or explanations

3. **Language**: Works best for English (stop words, tokenization)

## Future Enhancements

1. **Semantic Embeddings**: Use sentence transformers for better semantic similarity
2. **Option Comparison**: Also compare options to detect similar answer choices
3. **Topic Clustering**: Ensure questions cover different aspects/topics
4. **Difficulty Diversity**: Ensure mix of easy/medium/hard questions

## Testing

Test duplicate detection:

```python
from src.question_diversity import check_duplicate_questions

# Test 1: Clear duplicates
questions = [
    {"question": "What is Python?"},
    {"question": "What does Python mean?"},
]
is_diverse, warnings, unique = check_duplicate_questions(questions)
assert not is_diverse
assert len(unique) == 1

# Test 2: Different questions
questions = [
    {"question": "What is Python?"},
    {"question": "How do you write a for loop in Python?"},
]
is_diverse, warnings, unique = check_duplicate_questions(questions)
assert is_diverse
assert len(unique) == 2
```

## Dependencies

- `sklearn` (scikit-learn) for TF-IDF and cosine similarity
- Falls back gracefully if sklearn not available

