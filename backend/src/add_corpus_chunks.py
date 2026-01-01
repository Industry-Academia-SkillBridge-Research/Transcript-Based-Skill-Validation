"""
add_corpus_chunks.py

Helper script to add new chunks to the skill corpus.

Usage:
    python src/add_corpus_chunks.py --skill "SQL" --text "Your chunk text here..." --source "IT1010 - Database Systems"
    
    Or interactively:
    python src/add_corpus_chunks.py
"""

import argparse
import os
import pandas as pd
from typing import List, Dict
import re


def count_words(text: str) -> int:
    """Count words in text."""
    return len(re.findall(r'\b\w+\b', text))


def get_next_chunk_id(df: pd.DataFrame, skill: str) -> str:
    """Get the next chunk ID for a skill."""
    if df.empty:
        return f"{skill.lower().replace(' ', '_')}_chunk_001"
    
    skill_chunks = df[df["Skill"].str.lower() == skill.lower()]
    if skill_chunks.empty:
        return f"{skill.lower().replace(' ', '_')}_chunk_001"
    
    # Extract chunk numbers
    chunk_ids = skill_chunks["ChunkID"].astype(str)
    max_num = 0
    for chunk_id in chunk_ids:
        match = re.search(r'_chunk_(\d+)', chunk_id)
        if match:
            num = int(match.group(1))
            max_num = max(max_num, num)
    
    next_num = max_num + 1
    skill_key = skill.lower().replace(' ', '_').replace('-', '_')
    return f"{skill_key}_chunk_{next_num:03d}"


def split_large_text(text: str, max_words: int = 300) -> List[str]:
    """Split text into chunks if it's too large."""
    word_count = count_words(text)
    if word_count <= max_words:
        return [text]
    
    # Split by sentences
    sentences = re.split(r'[.!?]+\s+', text)
    chunks = []
    current_chunk = []
    current_words = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        sent_words = count_words(sentence)
        if current_words + sent_words > max_words and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_words = sent_words
        else:
            current_chunk.append(sentence)
            current_words += sent_words
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


def add_chunks_to_corpus(
    skill: str,
    text: str,
    source: str = "",
    corpus_path: str = "content/skill_corpus.csv"
):
    """Add new chunks to the corpus."""
    
    # Load existing corpus or create new
    if os.path.exists(corpus_path):
        df = pd.read_csv(corpus_path)
        if df.empty:
            df = pd.DataFrame(columns=["Skill", "ChunkID", "Text", "Source"])
    else:
        df = pd.DataFrame(columns=["Skill", "ChunkID", "Text", "Source"])
        os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
    
    # Split text if too large
    text_chunks = split_large_text(text, max_words=300)
    
    new_rows = []
    for chunk_text in text_chunks:
        chunk_id = get_next_chunk_id(df, skill)
        
        word_count = count_words(chunk_text)
        if word_count < 100:
            print(f"⚠️  Warning: Chunk '{chunk_id}' has only {word_count} words (recommended: 150-300)")
        elif word_count > 400:
            print(f"⚠️  Warning: Chunk '{chunk_id}' has {word_count} words (recommended: 150-300)")
        
        new_rows.append({
            "Skill": skill,
            "ChunkID": chunk_id,
            "Text": chunk_text.strip(),
            "Source": source.strip()
        })
    
    # Append new rows
    new_df = pd.DataFrame(new_rows)
    df = pd.concat([df, new_df], ignore_index=True)
    
    # Save
    df.to_csv(corpus_path, index=False)
    
    print(f"✅ Added {len(new_rows)} chunk(s) for skill '{skill}'")
    for row in new_rows:
        word_count = count_words(row["Text"])
        print(f"   - {row['ChunkID']}: {word_count} words")
    
    return df


def interactive_mode():
    """Run in interactive mode."""
    print("=" * 60)
    print("Add Chunks to Skill Corpus")
    print("=" * 60)
    print()
    
    skill = input("Skill name: ").strip()
    if not skill:
        print("❌ Skill name is required")
        return
    
    print("\nEnter chunk text (150-300 words recommended):")
    print("(Press Enter twice to finish, or Ctrl+C to cancel)")
    lines = []
    while True:
        try:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        except KeyboardInterrupt:
            print("\n❌ Cancelled")
            return
    
    text = '\n'.join(lines).strip()
    if not text:
        print("❌ No text provided")
        return
    
    source = input("\nSource (optional, e.g., 'IT1010 - Database Systems'): ").strip()
    
    print(f"\nAdding chunk(s) for skill: {skill}")
    add_chunks_to_corpus(skill, text, source)


def main():
    parser = argparse.ArgumentParser(
        description="Add chunks to the skill corpus for RAG-based quiz generation"
    )
    parser.add_argument(
        "--skill",
        type=str,
        help="Skill name (e.g., 'SQL', 'Machine Learning')"
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Chunk text content (150-300 words recommended)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="",
        help="Source information (optional, e.g., 'IT1010 - Database Systems', 'Lecture 5', 'Textbook Chapter 3')"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Read text from file instead of --text argument"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="content/skill_corpus.csv",
        help="Path to skill corpus CSV file (default: content/skill_corpus.csv)"
    )
    
    args = parser.parse_args()
    
    # Interactive mode if no arguments
    if not args.skill and not args.text and not args.file:
        interactive_mode()
        return
    
    if not args.skill:
        print("❌ --skill is required")
        return
    
    # Get text from file or argument
    if args.file:
        if not os.path.exists(args.file):
            print(f"❌ File not found: {args.file}")
            return
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        print("❌ Either --text or --file is required")
        return
    
    add_chunks_to_corpus(args.skill, text, args.source or "", args.corpus)


if __name__ == "__main__":
    main()

