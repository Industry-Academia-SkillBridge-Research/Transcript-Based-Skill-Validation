"""
build_skill_corpus_chunks.py

Purpose:
    Build a chunked skill knowledge corpus for RAG-based quiz generation.
    
    The corpus should have:
    - Skill: skill name
    - ChunkID: unique identifier for each chunk (e.g., "sql_chunk_001")
    - Text: chunk content (150-300 words per chunk)
    - Source: optional source (module name, lecture note, textbook section)
    
    Key requirement: Multiple chunks per skill (not one huge paragraph)
"""

import os
import re
import pandas as pd
from typing import List, Dict, Optional


def count_words(text: str) -> int:
    """Count words in text."""
    return len(re.findall(r'\b\w+\b', text))


def split_into_chunks(text: str, min_words: int = 150, max_words: int = 300, overlap: int = 50) -> List[str]:
    """
    Split text into chunks of 150-300 words with optional overlap.
    
    Args:
        text: Input text to chunk
        min_words: Minimum words per chunk
        max_words: Maximum words per chunk
        overlap: Number of words to overlap between chunks (for context preservation)
    
    Returns:
        List of text chunks
    """
    # Clean and split into sentences
    sentences = re.split(r'[.!?]+\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return []
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_words = count_words(sentence)
        
        # If adding this sentence would exceed max_words, save current chunk and start new one
        if current_word_count + sentence_words > max_words and current_word_count >= min_words:
            # Save current chunk
            chunk_text = ' '.join(current_chunk)
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
            
            # Start new chunk with overlap (last few sentences)
            if overlap > 0 and len(current_chunk) > 0:
                # Take last sentences to maintain context
                overlap_sentences = []
                overlap_count = 0
                for j in range(len(current_chunk) - 1, -1, -1):
                    sent = current_chunk[j]
                    sent_words = count_words(sent)
                    if overlap_count + sent_words <= overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_count += sent_words
                    else:
                        break
                current_chunk = overlap_sentences
                current_word_count = overlap_count
            else:
                current_chunk = []
                current_word_count = 0
        
        # Add sentence to current chunk
        current_chunk.append(sentence)
        current_word_count += sentence_words
        i += 1
    
    # Add remaining chunk
    if current_chunk and current_word_count >= min_words:
        chunk_text = ' '.join(current_chunk)
        if chunk_text.strip():
            chunks.append(chunk_text.strip())
    elif current_chunk:
        # If last chunk is too short, merge with previous
        if chunks:
            chunks[-1] = chunks[-1] + ' ' + ' '.join(current_chunk)
        else:
            chunks.append(' '.join(current_chunk))
    
    return chunks


def load_knowledge_base_files(kb_dir: str = "knowledge_base") -> Dict[str, str]:
    """
    Load all .txt files from knowledge_base directory.
    
    Returns:
        Dictionary mapping skill names to their full text content
    """
    kb_content = {}
    
    if not os.path.exists(kb_dir):
        print(f"Warning: Knowledge base directory '{kb_dir}' not found.")
        return kb_content
    
    for filename in os.listdir(kb_dir):
        if filename.lower().endswith('.txt'):
            skill_name = filename[:-4]  # Remove .txt extension
            filepath = os.path.join(kb_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read().strip()
                    if content:
                        kb_content[skill_name] = content
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
    
    return kb_content


def create_chunked_corpus(
    skill_name: str,
    text_content: str,
    source: Optional[str] = None,
    min_words: int = 150,
    max_words: int = 300
) -> List[Dict[str, str]]:
    """
    Create chunked corpus entries for a skill.
    
    Returns:
        List of dictionaries with Skill, ChunkID, Text, Source
    """
    chunks = split_into_chunks(text_content, min_words=min_words, max_words=max_words)
    
    entries = []
    for idx, chunk_text in enumerate(chunks):
        chunk_id = f"{skill_name.lower()}_chunk_{idx+1:03d}"
        entries.append({
            "Skill": skill_name,
            "ChunkID": chunk_id,
            "Text": chunk_text,
            "Source": source or ""
        })
    
    return entries


def load_existing_corpus(corpus_path: str = "content/skill_corpus.csv") -> pd.DataFrame:
    """
    Load existing skill_corpus.csv and convert to chunked format if needed.
    """
    if not os.path.exists(corpus_path):
        return pd.DataFrame()
    
    df = pd.read_csv(corpus_path)
    
    # If already has ChunkID column, return as is (might need re-chunking)
    if "ChunkID" in df.columns:
        return df
    
    # Convert old format (Skill, SourceType, SourceName, Content) to new format
    entries = []
    for _, row in df.iterrows():
        skill = str(row.get("Skill", "")).strip()
        content = str(row.get("Content", "")).strip()
        source_name = str(row.get("SourceName", "")).strip()
        source_type = str(row.get("SourceType", "")).strip()
        
        if not skill or not content:
            continue
        
        # Combine source info
        source = f"{source_type}: {source_name}".strip(": ").strip() if source_name else source_type
        
        # Create chunks
        chunks = create_chunked_corpus(skill, content, source=source)
        entries.extend(chunks)
    
    if entries:
        return pd.DataFrame(entries)
    return pd.DataFrame(columns=["Skill", "ChunkID", "Text", "Source"])


def main():
    """
    Main function to build chunked skill corpus.
    
    Sources:
    1. Existing knowledge_base/*.txt files
    2. Existing content/skill_corpus.csv (if any)
    """
    output_path = "content/skill_corpus.csv"
    
    print("Building chunked skill corpus...")
    print("=" * 60)
    
    all_entries = []
    
    # 1. Load from knowledge_base/*.txt files
    print("\n[1] Loading from knowledge_base/*.txt files...")
    kb_content = load_knowledge_base_files("knowledge_base")
    
    for skill_name, content in kb_content.items():
        print(f"  Processing: {skill_name} ({count_words(content)} words)")
        chunks = create_chunked_corpus(
            skill_name,
            content,
            source=f"knowledge_base/{skill_name}.txt"
        )
        all_entries.extend(chunks)
        print(f"    Created {len(chunks)} chunks")
    
    # 2. Load and convert existing CSV (if exists and not already chunked)
    existing_path = "content/skill_corpus.csv"
    if os.path.exists(existing_path):
        print("\n[2] Processing existing skill_corpus.csv...")
        existing_df = load_existing_corpus(existing_path)
        
        if not existing_df.empty:
            if "ChunkID" not in existing_df.columns:
                # Convert to chunks
                print("  Converting existing corpus to chunks...")
                existing_entries = existing_df.to_dict('records')
                all_entries.extend(existing_entries)
            else:
                print("  Existing corpus already has chunks, skipping conversion")
    
    if not all_entries:
        print("\n⚠️  No content found. Creating template...")
        print("   Please add content to:")
        print("   - knowledge_base/*.txt files, OR")
        print("   - content/skill_corpus.csv with columns: Skill, Content, Source")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_entries)
    
    # Ensure columns are in correct order
    df = df[["Skill", "ChunkID", "Text", "Source"]]
    
    # Sort by Skill, then ChunkID
    df = df.sort_values(["Skill", "ChunkID"]).reset_index(drop=True)
    
    # Validate chunks
    print("\n[3] Validating chunks...")
    word_counts = df["Text"].apply(count_words)
    print(f"  Total chunks: {len(df)}")
    print(f"  Average words per chunk: {word_counts.mean():.1f}")
    print(f"  Min words: {word_counts.min()}, Max words: {word_counts.max()}")
    
    # Check for chunks outside recommended range
    too_short = (word_counts < 100).sum()
    too_long = (word_counts > 400).sum()
    if too_short > 0:
        print(f"  ⚠️  {too_short} chunks have < 100 words (may be too short)")
    if too_long > 0:
        print(f"  ⚠️  {too_long} chunks have > 400 words (may be too long)")
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Saved {len(df)} chunks to {output_path}")
    print(f"\nSkills covered: {df['Skill'].nunique()}")
    print("\nSample chunks:")
    print(df[["Skill", "ChunkID", "Source"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()

