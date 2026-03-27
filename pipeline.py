import asyncio
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Global instances for lazy initialization
_vector_store = None
_keyword_search = None
_hybrid_search = None
_reranker = None
_llm = None


def setup_production_data(base_url: str = os.getenv("BASE_URL", "https://venerable-gelato-283258.netlify.app/")):
    """
    Performs the full production-ready ingestion process: 
    Crawl -> Clean -> Chunk -> Vector Embed.
    """
    print(f"🚀 Starting Production Ingestion for: {base_url}")
    
    # 1. Crawl the website
    from crawler import crawl as run_crawl
    raw_data = run_crawl(base_url)
    print(f"📄 Crawled {len(raw_data)} pages")
    
    if not raw_data:
        print("❌ No data crawled! Check if the website is running.")
        return
    
    # 2. Clean the crawled data
    from cleaner import TextCleaner
    cleaner = TextCleaner()
    cleaned_data = cleaner.clean_data(raw_data)
    print(f"🧹 Cleaned {len(cleaned_data)} pages")
    
    with open("cleaned_data.json", "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
    
    if not cleaned_data:
        print("❌ No data after cleaning! Cleaner might be too aggressive.")
        return
    
    # 3. Chunk the cleaned data
    from chunker import TextChunker
    chunker = TextChunker()
    chunks = chunker.create_chunks(cleaned_data, is_web=True)
    print(f"✂️ Created {len(chunks)} chunks")
    
    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    if not chunks:
        print("❌ No chunks created!")
        return
    
    # 4. Generate embeddings
    from vector_store import VectorStore
    vector_store = VectorStore()
    vector_store.create_embeddings_from_chunks(chunks, "embeddings.json")
    
    print("✅ Ingestion complete! Data is now ready for production search.")


def _initialize_components():
    """Initializes search and LLM components if they haven't been loaded yet."""
    global _vector_store, _keyword_search, _hybrid_search, _reranker, _llm
    
    if _llm is None:
        # Auto-run ingestion if data files are missing
        if not os.path.exists("embeddings.json") or not os.path.exists("chunks.json"):
            print("📁 Data files not found. Running ingestion...")
            setup_production_data()

        # Ensure data files exist before attempting to load
        if os.path.exists("embeddings.json") and os.path.exists("chunks.json"):
            print("🚀 Initializing Pipeline Components...")
            
            from vector_store import VectorStore
            from keyword_search import KeywordSearch
            from hybrid_search import HybridSearch
            from reRanker import Reranker
            from llm import LLM
            
            _vector_store = VectorStore()
            _vector_store.load_embeddings("embeddings.json")
            
            _keyword_search = KeywordSearch("chunks.json")
            _hybrid_search = HybridSearch(_vector_store, _keyword_search)
            _reranker = Reranker()
            _llm = LLM()
            print("✅ Pipeline Components Ready!")
        else:
            print("❌ Failed to create data files during ingestion.")


def run_pipeline(query: str, top_k: int = 5):
    """Main pipeline: Search -> Rerank -> Generate Answer"""
    _initialize_components()

    if _llm is None:
        return {
            "answer": "The system is still initializing or missing data. Please ensure ingestion is complete.",
            "sources": [],
            "intent": "system"
        }

    # Step 1: Hybrid Search
    results = _hybrid_search.search(query, top_k=top_k * 2)
    
    if not results:
        return {
            "answer": "I couldn't find any relevant information for your query.",
            "sources": [],
            "intent": "chat"
        }

    # Step 2: Rerank results
    ranked = _reranker.rerank(query, results, top_k=top_k)

    # Step 3: Prepare context
    context = "\n\n".join([f"[Source: {r.get('url', 'unknown')}]\n{r['text']}" for r in ranked])

    # Step 4: Generate LLM Answer
    answer = _llm.generate(query=query, context=context, chunks=ranked)

    return {
        "answer": answer,
        "sources": list(set([r.get("url", "unknown") for r in ranked])),
        "intent": "chat"
    }


def force_reindex(base_url: str = "https://venerable-gelato-283258.netlify.app/"):
    """Force a complete re-crawl and reindex of the website."""
    global _vector_store, _keyword_search, _hybrid_search, _reranker, _llm
    
    # Reset components
    _vector_store = None
    _keyword_search = None
    _hybrid_search = None
    _reranker = None
    _llm = None
    
    # Remove old data files
    for file in ["raw_data.json", "cleaned_data.json", "chunks.json", "embeddings.json"]:
        if os.path.exists(file):
            os.remove(file)
            print(f"🗑️ Removed {file}")
    
    # Run fresh ingestion
    setup_production_data(base_url)
    
    # Reinitialize components
    _initialize_components()


if __name__ == "__main__":
    # Test the pipeline
    force_reindex("https://venerable-gelato-283258.netlify.app/")
    result = run_pipeline("What is this website about?")
    print("\n" + "="*50)
    print("Answer:", result["answer"])
    print("Sources:", result["sources"])
