# RAG Retrieval for Quiz Generation

This document describes the RAG (Retrieval-Augmented Generation) retrieval system used for quiz generation.

## Overview

When a skill is selected for quiz generation, the system:

1. **Builds a query**: Skill name + 2-3 relevant keywords
2. **Retrieves top-k chunks** (k=3 to 6) from the skill corpus
3. **Stores retrieval results** with chunk IDs and concatenated text
4. **Uses retrieved context** to generate quiz questions

## Retrieval Methods

### 1. TF-IDF (Default, Baseline)

**Pros:**
- Fast and efficient
- No external dependencies beyond scikit-learn
- Good baseline performance
- Works well for keyword-based retrieval

**Usage:**
```python
from src.rag_retrieval import retrieve_skill_context

result = retrieve_skill_context(
    skill="SQL",
    corpus_df=corpus_df,
    method="tfidf",
    top_k=5
)
```

### 2. Embeddings + FAISS (Better Results)

**Pros:**
- Better semantic understanding
- Captures meaning beyond keywords
- More accurate retrieval for related concepts
- Better for research/research-quality output

**Requirements:**
```bash
pip install sentence-transformers faiss-cpu
```

**Usage:**
```python
result = retrieve_skill_context(
    skill="SQL",
    corpus_df=corpus_df,
    method="embeddings",
    top_k=5,
    model_name="all-MiniLM-L6-v2"  # Optional
)
```

**Model Options:**
- `all-MiniLM-L6-v2` (default): Fast, good quality, 384 dimensions
- `all-mpnet-base-v2`: Higher quality, slower, 768 dimensions
- `all-MiniLM-L12-v2`: Balanced, 384 dimensions

## Query Building

The system automatically builds queries using:

1. **Skill name** (e.g., "SQL")
2. **2-3 keywords** from the skill keyword dictionary

### Skill Keyword Dictionary

Predefined keywords for common skills:

```python
SKILL_KEYWORDS = {
    "SQL": ["database", "query", "table", "join", "select", "schema"],
    "Python": ["function", "class", "object", "module", "package"],
    "Machine Learning": ["model", "training", "algorithm", "prediction"],
    # ... more skills
}
```

### Custom Keywords

You can provide custom keywords:

```python
from src.rag_retrieval import retrieve_skill_context

result = retrieve_skill_context(
    skill="Advanced SQL",
    corpus_df=corpus_df,
    query="SQL database query optimization indexing",  # Custom query
    top_k=5
)
```

### Automatic Keyword Extraction

If a skill is not in the dictionary, keywords are automatically extracted from the skill name:
- Split on separators (`&`, `-`, `,`, spaces)
- Remove stop words
- Use top keywords

## Retrieval Output

The retrieval function returns a dictionary:

```python
{
    "skill": "SQL",
    "query": "SQL database query table",
    "retrieved_chunk_ids": [
        "sql_chunk_001",
        "sql_chunk_002",
        "sql_chunk_005"
    ],
    "retrieved_text": "SQL is a database language...\n\n---\n\nJOIN operations...",
    "chunks": [
        {
            "chunk_id": "sql_chunk_001",
            "text": "SQL is a database language...",
            "source": "IT1010 - Database Systems"
        },
        # ... more chunks
    ],
    "method": "tfidf"  # or "embeddings"
}
```

## Storing Retrieval Results

Retrieval results are automatically stored in `output/retrieval_results.csv`:

```csv
skill,query,retrieved_chunk_ids,num_chunks,method
SQL,"SQL database query table","sql_chunk_001,sql_chunk_002,sql_chunk_005",3,tfidf
```

## Integration with Quiz Generation

The retrieval system is integrated into `quiz_generation_rag.py`:

```python
# In generate_mcqs_for_plan_row()
retrieval_result = retrieve_skill_context(
    skill=skill,
    corpus_df=corpus_df,
    method="tfidf",  # or "embeddings"
    top_k=5
)

context = retrieval_result["retrieved_text"]
retrieved_chunk_ids = retrieval_result["retrieved_chunk_ids"]
```

The retrieved context is then used in the LLM prompt for question generation.

## Configuration

### Environment Variable

Set retrieval method via environment variable:

```bash
export RAG_RETRIEVAL_METHOD=embeddings  # or "tfidf"
python src/quiz_generation_rag.py
```

### In Code

```python
# Default: tfidf
result = retrieve_skill_context(skill, corpus_df)

# Explicit: embeddings
result = retrieve_skill_context(skill, corpus_df, method="embeddings")

# With custom options
result = retrieve_skill_context(
    skill,
    corpus_df,
    method="embeddings",
    model_name="all-mpnet-base-v2",
    top_k=6
)
```

## Performance Comparison

### TF-IDF
- **Speed**: Very fast (~ms per retrieval)
- **Memory**: Low
- **Accuracy**: Good for keyword matches
- **Best for**: Production systems, large corpora

### Embeddings
- **Speed**: Slower (requires model loading, ~100-500ms per retrieval)
- **Memory**: Higher (model + embeddings in memory)
- **Accuracy**: Better semantic understanding
- **Best for**: Research, higher quality requirements

## Best Practices

1. **Use TF-IDF for production** unless you need better semantic understanding
2. **Use embeddings for research** or when keyword matching isn't sufficient
3. **Set top_k=3-6** for optimal results (too few = incomplete context, too many = noise)
4. **Monitor retrieval results** in `output/retrieval_results.csv`
5. **Adjust keywords** in `SKILL_KEYWORDS` dictionary for better queries

## Troubleshooting

### "Embeddings not available, falling back to TF-IDF"
- Install dependencies: `pip install sentence-transformers faiss-cpu`
- Check if CUDA is available for GPU acceleration

### "No chunks found for skill"
- Check if skill name matches corpus exactly (case-sensitive)
- Verify corpus has chunks for the skill
- Try custom query with related keywords

### Poor retrieval quality
- Increase top_k (try 6 instead of 5)
- Add more keywords to SKILL_KEYWORDS
- Try embeddings method for better semantics
- Verify corpus chunks are well-structured (150-300 words)

## File Locations

- **Retrieval module**: `backend/src/rag_retrieval.py`
- **Quiz generation**: `backend/src/quiz_generation_rag.py`
- **Retrieval results**: `backend/output/retrieval_results.csv`
- **Corpus**: `backend/content/skill_corpus.csv`

