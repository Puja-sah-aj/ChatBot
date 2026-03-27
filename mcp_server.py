# mcp_server.py
from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import torch

from vector_store import VectorStore
from keyword_search import KeywordSearch
from hybrid_search import HybridSearch
from reRanker import Reranker
from llm import LLM, QueryIntent

faiss.omp_set_num_threads(1)
torch.set_num_threads(1)

app = FastAPI()

vector_store = None
keyword_search = None
hybrid_search = None
reranker = None
llm = None


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5   # ✅ Increased from 3 → 5 for more context


@app.on_event("startup")
def startup_event():
    global vector_store, keyword_search, hybrid_search, reranker, llm
    print("🚀 Loading system...")

    vector_store = VectorStore()
    vector_store.load_embeddings("embeddings.json")

    keyword_search = KeywordSearch("chunks.json")
    hybrid_search = HybridSearch(vector_store, keyword_search)
    reranker = Reranker()
    llm = LLM()

    print("✅ Ready!")


@app.post("/query")
def query_endpoint(payload: QueryRequest):
    query = payload.query.strip()
    intent = QueryIntent.detect(query)  # ✅ Detect intent first

    # ✅ Handle greetings
    if intent == "greeting":
        return {
            "answer": "Hello! 👋 I'm your real estate assistant. I can help you:\n"
                      "• 🏠 Find properties, villas, apartments\n"
                      "• 💰 Get pricing information\n"
                      "• 📍 Find properties by location\n"
                      "• 📞 Get contact details\n\n"
                      "What are you looking for today?"
        }

    # ✅ For list queries, increase top_k
    top_k = payload.top_k
    if intent == "list":
        top_k = max(top_k, 8)  # Get more results for lists

    # 🔍 Hybrid Search
    results = hybrid_search.search(query, top_k=top_k)
    if not results:
        return {"answer": "I couldn't find relevant information. Please try rephrasing or contact our team."}

    # 🔁 Rerank
    reranked = reranker.rerank(query, results, top_k=min(top_k, 5))
    if not reranked:
        return {"answer": "No useful content found. Please contact our team directly."}

    # 🧠 Build context
    context = " ".join([r["text"] for r in reranked])

    # ✅ Pass BOTH context string AND chunks (for source info)
    answer = llm.generate(query=query, context=context, chunks=reranked)

    return {
        "answer": answer,
        "intent": intent,          # ✅ Useful for frontend formatting
        "sources": [r["url"] for r in reranked[:3]]  # ✅ Source attribution
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mcp_server:app", host="0.0.0.0", port=8000, reload=False)
