"""
rag_retrieval.py

RAG Retrieval module for quiz generation.

Features:
- Keyword expansion (skill name + 2-3 keywords)
- TF-IDF retrieval (baseline, fast)
- Embeddings + FAISS retrieval (better results)
- Structured output with chunk IDs
"""

import os
import re
from typing import List, Dict, Optional, Tuple
import pandas as pd

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


# -----------------------------
# Skill Keyword Dictionary
# -----------------------------

SKILL_KEYWORDS = {
    # SQL & Databases
    "SQL": ["database", "query", "table", "join", "select", "schema"],
    "Database": ["sql", "table", "schema", "query", "normalization", "index"],
    "Database Design": ["erd", "entity", "relationship", "normalization", "schema"],
    
    # Programming Languages
    "Python": ["function", "class", "object", "module", "package", "syntax"],
    "Java": ["class", "object", "inheritance", "polymorphism", "interface", "package"],
    "C++": ["pointer", "memory", "class", "template", "inheritance", "stl"],
    "C": ["pointer", "array", "function", "struct", "memory", "compiler"],
    "JavaScript": ["function", "object", "dom", "async", "promise", "es6"],
    
    # Machine Learning
    "Machine Learning": ["model", "training", "algorithm", "prediction", "feature", "supervised"],
    "Deep Learning": ["neural network", "backpropagation", "gradient", "layer", "activation"],
    "Data Science": ["analysis", "statistics", "visualization", "pandas", "numpy"],
    
    # Web Development
    "Web Development": ["html", "css", "javascript", "api", "rest", "http"],
    "Frontend": ["html", "css", "javascript", "react", "dom", "ui"],
    "Backend": ["server", "api", "database", "rest", "http", "authentication"],
    
    # Statistics
    "Statistics": ["mean", "variance", "distribution", "hypothesis", "test", "probability"],
    "Data Visualization": ["chart", "graph", "plot", "dashboard", "visualization"],
    
    # Software Engineering
    "Software Engineering": ["design pattern", "uml", "testing", "agile", "scrum"],
    "Agile": ["sprint", "scrum", "backlog", "iteration", "standup"],
    
    # Networking
    "Networking": ["tcp", "ip", "protocol", "router", "switch", "subnet"],
    "Operating Systems": ["process", "thread", "memory", "scheduling", "file system"],
    
    # Security
    "Security": ["encryption", "authentication", "authorization", "vulnerability", "firewall"],
}

# Fallback: extract keywords from skill name if not in dictionary
def extract_keywords_from_skill(skill: str) -> List[str]:
    """Extract meaningful keywords from skill name."""
    # Split on common separators
    parts = re.split(r'[&\-,\s]+', skill.lower())
    # Filter out common stop words
    stop_words = {"and", "or", "the", "of", "for", "with", "in", "on", "at", "to", "a", "an"}
    keywords = [p.strip() for p in parts if p.strip() and p.strip() not in stop_words]
    return keywords[:5]  # Limit to 5 keywords


def build_query(skill: str, keywords: Optional[List[str]] = None) -> str:
    """
    Build retrieval query from skill name and 2-3 keywords.
    
    Query format: skill name + 2-3 keywords (from mapping or dictionary)
    
    Args:
        skill: Skill name
        keywords: Optional list of keywords. If None, will look up or extract.
    
    Returns:
        Query string combining skill name and 2-3 keywords
    """
    # Get keywords from dictionary or extract from skill name
    if keywords is None:
        # Try exact match first
        skill_key = skill.title()  # Normalize case
        if skill_key in SKILL_KEYWORDS:
            keywords = SKILL_KEYWORDS[skill_key]
        else:
            # Try case-insensitive match
            skill_lower = skill.lower()
            for key, values in SKILL_KEYWORDS.items():
                if key.lower() == skill_lower:
                    keywords = values
                    break
            else:
                # Fallback: extract from skill name
                keywords = extract_keywords_from_skill(skill)
    
    # Combine skill name with 2-3 top keywords (as specified: 2-3 keywords)
    query_parts = [skill]
    query_parts.extend(keywords[:3])  # Add top 3 keywords (will use 2-3)
    
    return " ".join(query_parts)


# -----------------------------
# TF-IDF Retrieval (Baseline)
# -----------------------------

class TFIDFRetriever:
    """TF-IDF based retriever for RAG."""
    
    def __init__(self, corpus_df: pd.DataFrame):
        """
        Initialize TF-IDF retriever.
        
        Args:
            corpus_df: DataFrame with columns: Skill, ChunkID, Content (or Text)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for TF-IDF retrieval")
        
        self.corpus_df = corpus_df.copy()
        # Ensure Content column exists (rename Text if needed)
        if "Content" not in self.corpus_df.columns and "Text" in self.corpus_df.columns:
            self.corpus_df["Content"] = self.corpus_df["Text"]
        
        # Build TF-IDF index
        contents = self.corpus_df["Content"].astype(str).tolist()
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_df=0.95,
            min_df=1,
            ngram_range=(1, 2),
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(contents)
    
    def retrieve(
        self,
        skill: str,
        query: Optional[str] = None,
        top_k: int = 5
    ) -> Dict[str, any]:
        """
        Retrieve top-k chunks for a skill.
        
        Args:
            skill: Skill name
            query: Optional query string. If None, will build from skill + keywords.
            top_k: Number of chunks to retrieve (default 5, range 3-6 recommended)
        
        Returns:
            Dictionary with:
            - skill: Skill name
            - retrieved_chunk_ids: List of chunk IDs
            - retrieved_text: Concatenated chunk texts
            - chunks: List of dicts with chunk_id and text
        """
        if query is None:
            query = build_query(skill)
        
        top_k = max(3, min(6, top_k))  # Clamp to 3-6 range
        
        # Filter corpus by skill (exact or partial match)
        mask = self.corpus_df["Skill"].fillna("").str.contains(
            skill, case=False, na=False, regex=False
        )
        filtered_df = self.corpus_df[mask].copy()
        
        if filtered_df.empty:
            # Fallback: use entire corpus
            filtered_df = self.corpus_df.copy()
        
        if filtered_df.empty:
            return {
                "skill": skill,
                "query": query,
                "retrieved_chunk_ids": [],
                "retrieved_text": "",
                "chunks": []
            }
        
        indices = filtered_df.index.tolist()
        sub_matrix = self.tfidf_matrix[indices, :]
        
        # Build query vector
        query_vector = self.vectorizer.transform([query])
        
        # Compute cosine similarity
        similarities = cosine_similarity(query_vector, sub_matrix).flatten()
        
        # Rank by similarity
        ranked = sorted(
            zip(indices, similarities),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Get top-k chunks
        top_indices = [idx for idx, _ in ranked[:top_k]]
        
        chunks = []
        chunk_ids = []
        texts = []
        
        for idx in top_indices:
            row = self.corpus_df.loc[idx]
            chunk_id = str(row.get("ChunkID", f"chunk_{idx}"))
            text = str(row["Content"]).strip()
            
            if text:
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": text,
                    "source": str(row.get("Source", ""))
                })
                chunk_ids.append(chunk_id)
                texts.append(text)
        
        retrieved_text = "\n\n---\n\n".join(texts)
        
        return {
            "skill": skill,
            "query": query,
            "retrieved_chunk_ids": chunk_ids,
            "retrieved_text": retrieved_text,
            "chunks": chunks,
            "method": "tfidf"
        }


# -----------------------------
# Embeddings + FAISS Retrieval
# -----------------------------

class EmbeddingRetriever:
    """Embeddings-based retriever using FAISS for fast similarity search."""
    
    def __init__(
        self,
        corpus_df: pd.DataFrame,
        model_name: str = "all-MiniLM-L6-v2",
        use_gpu: bool = False
    ):
        """
        Initialize embedding retriever.
        
        Args:
            corpus_df: DataFrame with columns: Skill, ChunkID, Content (or Text)
            model_name: Sentence transformer model name
            use_gpu: Whether to use GPU for embeddings
        """
        if not EMBEDDINGS_AVAILABLE:
            raise ImportError(
                "sentence-transformers and faiss are required for embedding retrieval.\n"
                "Install with: pip install sentence-transformers faiss-cpu"
            )
        
        if not NUMPY_AVAILABLE:
            raise ImportError("numpy is required for embedding retrieval")
        
        self.corpus_df = corpus_df.copy()
        # Ensure Content column exists
        if "Content" not in self.corpus_df.columns and "Text" in self.corpus_df.columns:
            self.corpus_df["Content"] = self.corpus_df["Text"]
        
        # Load sentence transformer model
        device = "cuda" if use_gpu else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        
        # Generate embeddings for all chunks
        print(f"Generating embeddings for {len(self.corpus_df)} chunks...")
        contents = self.corpus_df["Content"].astype(str).tolist()
        self.embeddings = self.model.encode(contents, show_progress_bar=True)
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance
        self.index.add(self.embeddings.astype('float32'))
    
    def retrieve(
        self,
        skill: str,
        query: Optional[str] = None,
        top_k: int = 5
    ) -> Dict[str, any]:
        """
        Retrieve top-k chunks using embeddings.
        
        Args:
            skill: Skill name
            query: Optional query string. If None, will build from skill + keywords.
            top_k: Number of chunks to retrieve (default 5, range 3-6 recommended)
        
        Returns:
            Dictionary with:
            - skill: Skill name
            - retrieved_chunk_ids: List of chunk IDs
            - retrieved_text: Concatenated chunk texts
            - chunks: List of dicts with chunk_id and text
        """
        if query is None:
            query = build_query(skill)
        
        top_k = max(3, min(6, top_k))  # Clamp to 3-6 range
        
        # Generate query embedding
        query_embedding = self.model.encode([query])
        
        # Search FAISS index
        distances, indices = self.index.search(
            query_embedding.astype('float32'),
            min(top_k * 2, len(self.corpus_df))  # Get more candidates
        )
        
        # Filter by skill match (if corpus is large enough)
        filtered_indices = []
        skill_mask = self.corpus_df["Skill"].fillna("").str.contains(
            skill, case=False, na=False, regex=False
        )
        
        for idx in indices[0]:
            idx_int = int(idx)
            if idx_int < len(self.corpus_df):
                # Prefer skill matches, but also include top results
                if skill_mask.iloc[idx_int] or len(filtered_indices) < top_k:
                    filtered_indices.append(idx_int)
                if len(filtered_indices) >= top_k:
                    break
        
        # If no skill matches, use top results anyway
        if not filtered_indices:
            filtered_indices = [int(idx) for idx in indices[0][:top_k]]
        
        chunks = []
        chunk_ids = []
        texts = []
        
        for idx in filtered_indices[:top_k]:
            row = self.corpus_df.iloc[idx]
            chunk_id = str(row.get("ChunkID", f"chunk_{idx}"))
            text = str(row["Content"]).strip()
            
            if text:
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": text,
                    "source": str(row.get("Source", ""))
                })
                chunk_ids.append(chunk_id)
                texts.append(text)
        
        retrieved_text = "\n\n---\n\n".join(texts)
        
        return {
            "skill": skill,
            "query": query,
            "retrieved_chunk_ids": chunk_ids,
            "retrieved_text": retrieved_text,
            "chunks": chunks,
            "method": "embeddings"
        }


# -----------------------------
# Main Retrieval Function
# -----------------------------

def retrieve_skill_context(
    skill: str,
    corpus_df: pd.DataFrame,
    method: str = "tfidf",
    top_k: int = 5,
    query: Optional[str] = None,
    **kwargs
) -> Dict[str, any]:
    """
    Main function to retrieve context for a skill.
    
    Args:
        skill: Skill name
        corpus_df: Corpus DataFrame with Skill, ChunkID, Content columns
        method: Retrieval method - "tfidf" or "embeddings" (default: "tfidf")
        top_k: Number of chunks to retrieve (3-6 recommended)
        query: Optional query string. If None, will build from skill + keywords.
        **kwargs: Additional arguments for retriever initialization
    
    Returns:
        Dictionary with retrieval results:
        - skill: Skill name
        - query: Query used for retrieval
        - retrieved_chunk_ids: List of chunk IDs
        - retrieved_text: Concatenated chunk texts
        - chunks: List of dicts with chunk_id, text, source
        - method: Retrieval method used
    """
    if method.lower() == "embeddings" and EMBEDDINGS_AVAILABLE:
        retriever = EmbeddingRetriever(corpus_df, **kwargs)
    else:
        if method.lower() == "embeddings":
            print("Warning: Embeddings not available, falling back to TF-IDF")
        retriever = TFIDFRetriever(corpus_df)
    
    return retriever.retrieve(skill, query=query, top_k=top_k)


# -----------------------------
# Store Retrieval Results
# -----------------------------

def save_retrieval_results(
    retrieval_results: List[Dict[str, any]],
    output_path: str = "output/retrieval_results.csv"
):
    """
    Save retrieval results to CSV.
    
    Args:
        retrieval_results: List of retrieval result dictionaries
        output_path: Path to save CSV file
    """
    rows = []
    for result in retrieval_results:
        rows.append({
            "skill": result["skill"],
            "query": result["query"],
            "retrieved_chunk_ids": ",".join(result["retrieved_chunk_ids"]),
            "num_chunks": len(result["retrieved_chunk_ids"]),
            "method": result.get("method", "unknown")
        })
    
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(rows)} retrieval results to {output_path}")

