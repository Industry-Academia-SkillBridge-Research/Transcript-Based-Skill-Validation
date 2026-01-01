"""
test_rag_retrieval.py

Test script to demonstrate RAG retrieval functionality.
"""

import pandas as pd
from src.rag_retrieval import retrieve_skill_context, build_query, save_retrieval_results

def test_retrieval():
    """Test the RAG retrieval system."""
    
    # Load corpus
    corpus_path = "content/skill_corpus.csv"
    try:
        corpus_df = pd.read_csv(corpus_path)
        print(f"✅ Loaded corpus: {len(corpus_df)} chunks")
        print(f"   Skills: {corpus_df['Skill'].nunique()}")
    except FileNotFoundError:
        print(f"❌ Corpus file not found: {corpus_path}")
        print("   Run: python src/build_skill_corpus_chunks.py first")
        return
    
    # Test skills
    test_skills = ["SQL", "Python", "Machine Learning"]
    
    print("\n" + "="*60)
    print("Testing RAG Retrieval")
    print("="*60)
    
    all_results = []
    
    for skill in test_skills:
        print(f"\n📚 Skill: {skill}")
        print("-" * 60)
        
        # Build query
        query = build_query(skill)
        print(f"Query: {query}")
        
        # Test TF-IDF retrieval
        print("\n[TF-IDF Retrieval]")
        try:
            result_tfidf = retrieve_skill_context(
                skill=skill,
                corpus_df=corpus_df,
                method="tfidf",
                top_k=5
            )
            
            print(f"  ✅ Retrieved {len(result_tfidf['retrieved_chunk_ids'])} chunks")
            print(f"  Chunk IDs: {result_tfidf['retrieved_chunk_ids']}")
            print(f"  Text length: {len(result_tfidf['retrieved_text'])} characters")
            
            all_results.append(result_tfidf)
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        # Test Embeddings retrieval (if available)
        print("\n[Embeddings + FAISS Retrieval]")
        try:
            result_emb = retrieve_skill_context(
                skill=skill,
                corpus_df=corpus_df,
                method="embeddings",
                top_k=5
            )
            
            print(f"  ✅ Retrieved {len(result_emb['retrieved_chunk_ids'])} chunks")
            print(f"  Chunk IDs: {result_emb['retrieved_chunk_ids']}")
            print(f"  Text length: {len(result_emb['retrieved_text'])} characters")
            
        except ImportError:
            print("  ⚠️  Embeddings not available (install: pip install sentence-transformers faiss-cpu)")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Save results
    if all_results:
        save_retrieval_results(all_results, "output/test_retrieval_results.csv")
        print(f"\n✅ Saved {len(all_results)} retrieval results to output/test_retrieval_results.csv")
    
    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)


if __name__ == "__main__":
    test_retrieval()

