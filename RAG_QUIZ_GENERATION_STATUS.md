# RAG + Gemini Quiz Generation - Implementation Status

## ✅ REQUIREMENT 5: Quiz Generation Using RAG + Gemini (Anti-Hallucination)

**Goal**: Generate quiz questions using RAG (Retrieval-Augmented Generation) to ground Gemini in a knowledge corpus, preventing hallucinations.

**Research Importance**: This is where the system becomes "advanced" - instead of letting Gemini invent content, force it to use a skill knowledge corpus.

---

## Status: ✅ **FULLY IMPLEMENTED**

### ✅ Step 1: Skill Knowledge Corpus
**Status**: ✅ **COMPLETE**

**File**: `backend/content/skill_corpus.csv`

**Structure:**
- `Skill` - Skill name
- `SourceType` - Source of content (e.g., "module_outline", "textbook", "lecture")
- `SourceName` - Specific source identifier
- `Content` - Grounded text for that skill (chunked content, 150-300 words)

**Format**: CSV with chunked content (not one huge paragraph per skill)

**Files:**
- `backend/content/skill_corpus.csv` - Main corpus file
- `backend/src/build_skill_corpus_chunks.py` - Script to build chunked corpus

**Example:**
```csv
Skill,SourceType,SourceName,Content
SQL & Relational Databases,module_outline,IT2100,"SQL is a domain-specific language used for managing relational databases. It includes commands for creating tables, inserting data, querying with SELECT statements, and joining multiple tables..."
```

---

### ✅ Step 2: RAG Retrieval System
**Status**: ✅ **COMPLETE**

**Two Retrieval Methods Implemented:**

#### 2a. TF-IDF Retrieval (Baseline, Fast)
- ✅ Keyword-based retrieval using TF-IDF + cosine similarity
- ✅ Fast and efficient
- ✅ No external dependencies beyond scikit-learn
- ✅ Good baseline performance
- ✅ Default method

**Files:**
- `backend/src/rag_retrieval.py` - `TFIDFRetriever` class

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

#### 2b. Embeddings + FAISS Retrieval (Better Results)
- ✅ Semantic retrieval using sentence transformers
- ✅ FAISS index for fast similarity search
- ✅ Better semantic understanding (beyond keywords)
- ✅ More accurate retrieval for related concepts
- ✅ Better for research-quality output

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

**Files:**
- `backend/src/rag_retrieval.py` - `EmbeddingRetriever` class

#### Retrieval Output:
```python
{
    "skill": "SQL",
    "query": "SQL database query table join",
    "retrieved_chunk_ids": ["chunk_001", "chunk_002", ...],
    "retrieved_text": "Concatenated chunks...",
    "chunks": [
        {"chunk_id": "chunk_001", "text": "...", "source": "..."},
        ...
    ],
    "method": "tfidf" or "embeddings"
}
```

---

### ✅ Step 3: Query Construction with Keyword Expansion
**Status**: ✅ **COMPLETE**

**Query Format**: Skill name + 2-3 keywords

**Keyword Dictionary:**
- ✅ Predefined keywords for common skills (SQL, Python, Machine Learning, etc.)
- ✅ Fallback: Extract keywords from skill name if not in dictionary
- ✅ Query automatically built: `skill_name + keyword1 + keyword2 + keyword3`

**Examples:**
- SQL → "SQL database query table join select schema"
- Python → "Python function class object module package syntax"
- Machine Learning → "Machine Learning model training algorithm prediction feature supervised"

**Files:**
- `backend/src/rag_retrieval.py` - `SKILL_KEYWORDS` dictionary, `build_query()` function

---

### ✅ Step 4: Gemini Question Generation with Context
**Status**: ✅ **COMPLETE**

**Prompt Structure:**
```
You are an examiner generating multiple choice questions.

Create N multiple choice questions using ONLY the provided context.
Do not use outside knowledge. If the context is insufficient, return fewer questions.

SkillKey: {skill}
Difficulty: {difficulty}
NumberOfQuestions: {n}
Valid chunk IDs for citations: {chunk_ids}

QUESTION DIVERSITY REQUIREMENT:
- Do NOT repeat question types. Mix different question formats:
  * Definition questions: "What is X?"
  * Scenario-based questions: "In a situation where..."
  * Application questions: "How would you apply X to..."
  * Comparison questions: "What is the difference between X and Y?"

Context:
{retrieved_context}

Return ONLY valid JSON in this EXACT schema:
{
  "skill_key": "{skill}",
  "questions": [
    {
      "question": "text (10-500 characters)",
      "options": {"A":"...", "B":"...", "C":"...", "D":"..."},
      "answer": "A|B|C|D",
      "explanation": "text (minimum 20 characters)",
      "evidence": [
        {"chunk_id": "chunk_id_from_context", "quote": "relevant quote"}
      ]
    }
  ]
}

IMPORTANT RULES:
- All options (A, B, C, D) must be unique
- Answer must be exactly A, B, C, or D
- Question text: 10-500 characters
- Explanation: minimum 20 characters
- Evidence chunk_ids must match those in context
- Return ONLY JSON, no markdown, no explanation
```

**Key Features:**
- ✅ Forces Gemini to use ONLY provided context (anti-hallucination)
- ✅ Requires evidence citations with chunk IDs
- ✅ Validates citations match retrieved chunks
- ✅ Question diversity requirement (mix question types)
- ✅ Robust JSON validation and repair

**Files:**
- `backend/src/quiz_generation_gemini.py` - `generate_mcqs_from_context()` function
- `backend/src/quiz_validation.py` - Validation and repair logic

---

### ✅ Step 5: Question Output Format
**Status**: ✅ **COMPLETE**

**Generated Question Structure:**
- ✅ `QuestionID` - Unique identifier
- ✅ `SelectedSkill` - Original skill selected by user
- ✅ `Skill` - Skill used for generation (may differ due to mapping)
- ✅ `Difficulty` - Question difficulty
- ✅ `QuestionText` - Question text (10-500 characters)
- ✅ `OptionA`, `OptionB`, `OptionC`, `OptionD` - Four unique options
- ✅ `CorrectOption` - Correct answer (A/B/C/D)
- ✅ `Explanation` - Explanation (references retrieved context)
- ✅ `Source` - Source marker ("gemini" for RAG-generated, "question_bank" for fallback)

**Output Files:**

1. **Quiz Questions (with answers)** - `output/quizzes/{quiz_id}_questions.json`
   - Full questions with correct answers
   - Used by backend for scoring

2. **Quiz Metadata** - `output/quizzes/{quiz_id}_meta.json`
   - Quiz ID, student ID, selected skills
   - Time limit, difficulty, etc.

3. **Retrieval Results** (optional) - `output/retrieval_results.csv`
   - Skill, retrieved_chunk_ids, retrieval_query, method

**Note**: The batch processing script also saves to `output/quiz_questions_generated.csv` for batch runs.

**Files:**
- `backend/src/api/main.py` - Quiz generation endpoint
- `backend/src/quiz_generation_rag.py` - Batch generation script

---

### ✅ Step 6: Frontend Quiz Display
**Status**: ✅ **COMPLETE** (with minor enhancement needed for timer)

**Quiz Screen Features:**
- ✅ Questions displayed one at a time
- ✅ Questions grouped by selected skills (skill badge shown)
- ✅ Progress bar and question counter
- ✅ Options displayed as clickable cards
- ✅ Navigation (Previous/Next buttons)
- ✅ Question indicator dots
- ✅ Skill and difficulty badges per question

**Current Implementation:**
- ✅ Questions fetched and displayed
- ✅ Shuffled question order (client-side)
- ✅ Shuffled options per question (client-side)
- ⚠️ **Timer/Countdown**: Not yet fully implemented in QuizPage.jsx (needs enhancement)

**Files:**
- `frontend-react/src/components/QuizPage.jsx` - Main quiz display component
- `frontend-react/src/components/QuizResultPage.jsx` - Results display

**Enhancement Needed:**
- Timer countdown display (currently time_limit is received but not displayed)

---

## 📊 Generation Workflow

```
1. User selects skills (max 5)
   ↓
2. For each selected skill:
   a. Build query: skill + keywords
   b. Retrieve top-k chunks (k=3-6) from corpus
      - Method: TF-IDF (default) or Embeddings
   c. Concatenate retrieved chunks → context
   d. Send to Gemini with prompt:
      "Generate MCQs using ONLY this context"
   e. Validate JSON output:
      - Check schema, fields, option uniqueness
      - Validate citations match chunk IDs
      - Repair if needed (re-prompt Gemini)
   f. Store questions with:
      - QuestionText, Options A-D, CorrectOption
      - Explanation (references context)
      - Evidence citations
   ↓
3. Save questions to:
   - output/quizzes/{quiz_id}_questions.json
   - output/quizzes/{quiz_id}_meta.json
   ↓
4. Return to frontend (questions without answers)
   ↓
5. Frontend displays:
   - Timed quiz (timer enhancement needed)
   - Questions grouped by skill
   - Shuffled order and options
```

---

## 🔍 Anti-Hallucination Mechanisms

1. **Context-Only Prompting**:
   - Prompt explicitly states: "using ONLY the provided context"
   - No outside knowledge allowed

2. **Citation Validation**:
   - Requires evidence with chunk_ids
   - Validates chunk_ids match retrieved chunks
   - Rejects questions with invalid citations

3. **Context Insufficient Handling**:
   - If context insufficient, Gemini returns fewer questions
   - System validates minimum requirements

4. **Repair and Regeneration**:
   - If validation fails, attempts repair (re-prompt)
   - If repair fails, regenerates with different chunks or reduced difficulty

---

## ✅ Verification Checklist

- [x] Skill corpus exists (`content/skill_corpus.csv`)
- [x] Corpus has chunked content (not huge paragraphs)
- [x] TF-IDF retrieval implemented (baseline)
- [x] Embeddings + FAISS retrieval implemented (advanced)
- [x] Query construction with keyword expansion
- [x] Keyword dictionary for common skills
- [x] Retrieval returns top-k chunks (k=3-6)
- [x] Retrieval returns chunk IDs
- [x] Gemini prompt forces context-only usage
- [x] Questions generated with context
- [x] Explanations reference retrieved context
- [x] Citations validated against chunk IDs
- [x] JSON validation and repair
- [x] Questions saved to output files
- [x] Frontend displays quiz questions
- [x] Questions grouped by skill (badge shown)
- [x] Question order shuffled
- [x] Options shuffled per question
- [ ] Timer countdown displayed (needs enhancement)

---

## 📝 Example Generation Flow

**Input:**
- Selected Skill: "SQL & Relational Databases"
- Questions per skill: 3
- Difficulty: "Medium"

**Step 1: Query Construction**
- Skill: "SQL & Relational Databases"
- Keywords: ["database", "query", "table", "join", "select", "schema"]
- Query: "SQL & Relational Databases database query table join select schema"

**Step 2: Retrieval (TF-IDF)**
- Retrieved chunks: 5 chunks from corpus
- Chunk IDs: ["chunk_001", "chunk_002", "chunk_003", "chunk_004", "chunk_005"]
- Concatenated context: "SQL is a domain-specific language... [5000 chars]"

**Step 3: Gemini Generation**
- Prompt: "Generate 3 MCQs using ONLY this context..."
- Generated questions: 3 questions with explanations

**Step 4: Validation**
- JSON schema: ✓ Valid
- Options unique: ✓ All unique
- Citations valid: ✓ All chunk_ids match
- Question length: ✓ 10-500 chars
- Explanation length: ✓ ≥20 chars

**Step 5: Output**
```json
{
  "QuestionID": 100001,
  "SelectedSkill": "SQL & Relational Databases",
  "Skill": "SQL & Relational Databases",
  "Difficulty": "medium",
  "QuestionText": "What is the primary purpose of a JOIN operation in SQL?",
  "OptionA": "To combine rows from multiple tables",
  "OptionB": "To delete duplicate records",
  "OptionC": "To sort query results",
  "OptionD": "To create new tables",
  "CorrectOption": "A",
  "Explanation": "JOIN operations combine rows from multiple tables based on a related column, as described in chunk_002...",
  "Source": "gemini"
}
```

---

## ⚠️ Manual Setup Notes

### Required Setup:

1. **Build Skill Corpus**:
   ```bash
   cd backend
   python src/build_skill_corpus_chunks.py
   ```
   Output: `content/skill_corpus.csv`

2. **Gemini API Key**:
   Set environment variable:
   ```bash
   export GEMINI_API_KEY="your-api-key"
   ```
   Or add to `.env` file

3. **Optional: Enable Embeddings**:
   ```bash
   pip install sentence-transformers faiss-cpu
   ```
   Then set: `RAG_RETRIEVAL_METHOD=embeddings`

### Corpus Format:

**Required Columns:**
- `Skill` - Skill name
- `Content` - Chunked text (150-300 words recommended)

**Optional Columns:**
- `ChunkID` - Unique chunk identifier
- `Source` - Source name (module, textbook, etc.)
- `SourceType` - Type of source

---

## 🎯 Summary

**Status**: ✅ **COMPLETE** (with minor timer enhancement needed)

All requirements for Step 5 (RAG + Gemini Quiz Generation) are implemented:
- ✅ Skill knowledge corpus with chunked content
- ✅ RAG retrieval (TF-IDF and Embeddings + FAISS)
- ✅ Query construction with keyword expansion
- ✅ Gemini generation with context-only prompting
- ✅ Anti-hallucination mechanisms (citations, validation)
- ✅ Question output with all required fields
- ✅ Frontend quiz display (timer enhancement recommended)

**Enhancement Recommended**: Add visible timer countdown in QuizPage component to show remaining time during quiz.

