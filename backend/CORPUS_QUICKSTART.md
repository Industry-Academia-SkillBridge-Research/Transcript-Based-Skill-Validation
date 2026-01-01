# Quick Start: Building Skill Corpus

## Overview
The skill corpus is used by Gemini to generate quiz questions. It must have **multiple chunks per skill** (150-300 words each) for precise retrieval.

## Quick Setup

### Step 1: Build Corpus from Knowledge Base Files

If you have `.txt` files in `knowledge_base/` directory:

```bash
cd backend
python src/build_skill_corpus_chunks.py
```

This automatically:
- Loads all `.txt` files from `knowledge_base/`
- Chunks them into 150-300 word pieces
- Creates `content/skill_corpus.csv` with proper structure

### Step 2: Verify Corpus

Check the output:
- Total chunks created
- Average words per chunk
- Any warnings for chunks outside recommended size

### Step 3: Test Quiz Generation

Generate a quiz and verify questions are relevant to the skills.

## Adding New Chunks

### Interactive Mode
```bash
python src/add_corpus_chunks.py
```

### Command Line
```bash
python src/add_corpus_chunks.py \
    --skill "SQL" \
    --text "Your chunk text here (150-300 words)..." \
    --source "IT1010 - Database Systems"
```

### From File
```bash
python src/add_corpus_chunks.py \
    --skill "Machine Learning" \
    --file knowledge_base/ml_notes.txt \
    --source "IT3010 - ML Course"
```

## Corpus Format

The `content/skill_corpus.csv` file has these columns:

| Skill | ChunkID | Text | Source |
|-------|---------|------|--------|
| SQL | sql_chunk_001 | SQL is a database language... | IT1010 - Database Systems |
| SQL | sql_chunk_002 | JOIN operations allow... | IT1010 - Database Systems |
| Machine Learning | ml_chunk_001 | Machine learning is... | IT3010 - ML Course |

## Important Notes

✅ **DO**: 
- Create multiple chunks per skill (3-10 chunks recommended)
- Keep chunks between 150-300 words
- Make each chunk cover one concept/subtopic
- Include source information

❌ **DON'T**:
- Store one huge paragraph per skill
- Create chunks smaller than 100 words
- Create chunks larger than 400 words

## Troubleshooting

**Problem**: "No chunks found for skill"
- Check skill name matches exactly (case-sensitive)
- Run `build_skill_corpus_chunks.py` to rebuild

**Problem**: Questions not relevant
- Add more specific chunks for the skill
- Ensure chunks cover the skill comprehensively

For detailed documentation, see `docs/BUILD_SKILL_CORPUS.md`

