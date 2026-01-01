# RAG Retrieval Usage Guide

## Quick Start

### Basic Usage

```python
from src.rag_retrieval import retrieve_skill_context
import pandas as pd

# Load corpus
corpus_df = pd.read_csv("content/skill_corpus.csv")

# Retrieve context for a skill
result = retrieve_skill_context(
    skill="SQL",
    corpus_df=corpus_df,
    method="tfidf",  # or "embeddings"
    top_k=5  # k=3 to 6
)

# Access results
print(f"Skill: {result['skill']}")
print(f"Query: {result['query']}")  # skill + 2-3 keywords
print(f"Chunk IDs: {result['retrieved_chunk_ids']}")
print(f"Text: {result['retrieved_text']}")
```

## Output Format

The retrieval returns a dictionary with:

```python
{
    "skill": "SQL",                    # Skill name
    "query": "SQL database query table",  # skill + 2-3 keywords
    "retrieved_chunk_ids": [          # List of chunk IDs
        "sql_chunk_001",
        "sql_chunk_002",
        "sql_chunk_005"
    ],
    "retrieved_text": "SQL is a database language...\n\n---\n\nJOIN operations...",  # Concatenated chunks
    "chunks": [                       # Detailed chunk info
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

The system automatically stores retrieval results:

```python
from src.rag_retrieval import save_retrieval_results

results = [
    {
        "skill": "SQL",
        "query": "SQL database query",
        "retrieved_chunk_ids": ["sql_chunk_001", "sql_chunk_002"],
        "method": "tfidf"
    }
]

save_retrieval_results(results, "output/retrieval_results.csv")
```

Saved CSV format:
```csv
skill,query,retrieved_chunk_ids,num_chunks,method
SQL,"SQL database query table","sql_chunk_001,sql_chunk_002,sql_chunk_005",3,tfidf
```

## Testing

Run the test script:

```bash
python src/test_rag_retrieval.py
```

This will:
- Test retrieval for sample skills
- Show TF-IDF and Embeddings results
- Save test results to `output/test_retrieval_results.csv`

