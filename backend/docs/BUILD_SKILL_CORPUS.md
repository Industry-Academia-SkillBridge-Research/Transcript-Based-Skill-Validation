# Building the Skill Knowledge Base (Corpus)

This guide explains how to build and maintain the skill knowledge base used for RAG-based quiz generation.

## Overview

The skill corpus is a structured collection of knowledge chunks that Gemini uses to generate quiz questions. Each skill should have multiple small chunks (150-300 words) for precise retrieval.

## Corpus Structure

The corpus is stored in `content/skill_corpus.csv` with the following columns:

- **Skill**: Skill name (e.g., "SQL", "Machine Learning", "Database Design")
- **ChunkID**: Unique identifier for each chunk (e.g., `sql_chunk_001`, `sql_chunk_002`)
- **Text**: Chunk content (150-300 words recommended)
- **Source**: Optional source information (e.g., "IT1010 - Database Systems", "Lecture 5", "Textbook Chapter 3")

## Why Chunking Matters

**DO NOT** store one huge paragraph per skill. Instead:

- ✅ Break content into **multiple small chunks** (150-300 words each)
- ✅ Each chunk should cover a specific concept or subtopic
- ✅ Use overlap between chunks to preserve context
- ✅ This allows precise retrieval - only relevant chunks are used

### Example: SQL Skill

**Bad** (one huge chunk):
```
SQL is a database language... [500 words covering everything]
```

**Good** (multiple chunks):
```
Chunk 1: SQL Basics (SELECT, FROM, WHERE) - 200 words
Chunk 2: JOIN operations (INNER, LEFT, RIGHT) - 180 words
Chunk 3: Aggregation functions (COUNT, SUM, AVG) - 190 words
Chunk 4: Subqueries and nested queries - 220 words
```

## Building the Corpus

### Method 1: Automated Chunking from Knowledge Base Files

If you have text files in `knowledge_base/*.txt`:

```bash
python src/build_skill_corpus_chunks.py
```

This will:
1. Load all `.txt` files from `knowledge_base/` directory
2. Automatically chunk them into 150-300 word pieces
3. Create `content/skill_corpus.csv` with proper ChunkID values
4. Validate chunk sizes

### Method 2: Converting Existing Corpus

If you have an existing `skill_corpus.csv` with the old format (Skill, Content):

```bash
python src/build_skill_corpus_chunks.py
```

The script automatically detects and converts old format to chunked format.

### Method 3: Adding Chunks Manually

#### Using the helper script:

```bash
# Interactive mode
python src/add_corpus_chunks.py

# Or with arguments
python src/add_corpus_chunks.py \
    --skill "SQL" \
    --text "SQL is a database language used for managing relational databases..." \
    --source "IT1010 - Database Systems"
```

#### Or edit the CSV directly:

1. Open `content/skill_corpus.csv`
2. Add a new row with:
   - **Skill**: The skill name
   - **ChunkID**: `{skill_lowercase}_chunk_{number}` (e.g., `sql_chunk_001`)
   - **Text**: 150-300 words of content
   - **Source**: Optional source information

### Method 4: Adding Chunks from File

```bash
python src/add_corpus_chunks.py \
    --skill "Machine Learning" \
    --file knowledge_base/machine_learning.txt \
    --source "IT3010 - Machine Learning"
```

## Best Practices

### 1. Chunk Size
- **Target**: 150-300 words per chunk
- **Minimum**: 100 words (below this, context may be insufficient)
- **Maximum**: 400 words (above this, retrieval becomes less precise)

### 2. Chunk Content
- Each chunk should cover **one concept** or **one subtopic**
- Make chunks **self-contained** (readable on their own)
- Use **clear, concise language**
- Include **examples** when helpful

### 3. Source Information
- Include module codes (e.g., "IT1010 - Database Systems")
- Reference specific lectures or chapters
- Helps traceability and updating

### 4. Skill Naming
- Use consistent naming (matches skills in `course_skill_mapping.csv`)
- Consider case sensitivity (e.g., "SQL" vs "sql")

### 5. Multiple Chunks per Skill
- **Recommended**: 3-10 chunks per skill
- More chunks = better retrieval precision
- Different chunks can cover:
  - Basic concepts
  - Advanced topics
  - Practical examples
  - Common mistakes
  - Best practices

## Validation

After building the corpus, validate it:

```bash
python src/build_skill_corpus_chunks.py
```

This will show:
- Total number of chunks
- Average words per chunk
- Warnings for chunks outside recommended size range
- Skills covered

## Example Workflow

### Creating Corpus for a New Skill

1. **Prepare content**:
   ```
   Write comprehensive notes about the skill (e.g., SQL)
   Break it into logical sections
   ```

2. **Create chunks**:
   ```bash
   python src/add_corpus_chunks.py \
       --skill "SQL" \
       --file sql_notes.txt \
       --source "IT1010 - Database Systems"
   ```

3. **Verify chunks**:
   ```bash
   python src/build_skill_corpus_chunks.py
   ```

4. **Test quiz generation**:
   - Generate a quiz for the skill
   - Check if questions are relevant
   - Add more chunks if needed

## Troubleshooting

### "No chunks found for skill X"
- Check if skill name matches exactly (case-sensitive)
- Verify chunks exist in corpus
- Check CSV file encoding (should be UTF-8)

### "Generated questions are not relevant"
- Add more specific chunks
- Ensure chunks cover the skill comprehensively
- Check chunk size (may be too large/small)

### "Corpus file not found"
- Run `build_skill_corpus_chunks.py` first
- Check file path: `content/skill_corpus.csv`
- Ensure `content/` directory exists

## File Locations

- **Corpus file**: `backend/content/skill_corpus.csv`
- **Knowledge base**: `backend/knowledge_base/*.txt`
- **Build script**: `backend/src/build_skill_corpus_chunks.py`
- **Add chunks script**: `backend/src/add_corpus_chunks.py`

## Next Steps

After building the corpus:
1. Test quiz generation for different skills
2. Review generated questions for quality
3. Iteratively add/refine chunks based on results
4. Keep corpus updated as curriculum changes

