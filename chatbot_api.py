from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional, List
import os
import asyncio

import ingest
from pipeline import setup_production_data, run_pipeline


# ✅ Lifespan approach (modern replacement for @app.on_event("startup"))
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- SERVER STARTUP ----
    print("🚀 Server starting... Initializing system...")

    try:
        # 🔹 Run ingestion automatically
        print("🧩 Checking data files...")
        if not os.path.exists("embeddings.json") or not os.path.exists("chunks.json"):
            print("🪄 Running automatic ingestion...")
            setup_production_data()
        else:
            print("✅ Data files already exist — skipping ingestion.")

        # 🔹 Pre-initialize pipeline components (force load all modules)
        print("⚙️ Preloading pipeline components...")
        _ = run_pipeline("System warmup query")  # this will auto-initialize everything
        print("✅ Pipeline ready to serve queries!")

    except Exception as e:
        print(f"❌ Initialization error: {e}")

    # Yield control back to FastAPI
    yield

    # ---- SERVER SHUTDOWN ----
    print("🧹 Server shutting down — cleanup done!")


# Create FastAPI app with lifespan
app = FastAPI(title="Real Estate Chatbot API 🚀", lifespan=lifespan)

# -------- CORS --------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domain(s) in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------- MODELS --------
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


class ChatResponse(BaseModel):
    answer: str
    intent: Optional[str] = None
    sources: Optional[List[str]] = []
    formatted: bool = True


# -------- ROUTES --------
@app.get("/")
def home():
    return {"status": "running", "message": "Real Estate Chatbot API 🚀"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/ingest")
def run_ingestion():
    """Manual re-ingestion endpoint (optional)."""
    try:
        ingest.main()
        # Re-run setup again to reload vector and hybrid search
        setup_production_data()
        print("✅ Ingestion completed successfully and system reloaded.")
        return {"status": "Ingestion completed successfully ✅"}
    except Exception as e:
        return {"error": str(e)}


import asyncio

@app.post("/ask", response_model=ChatResponse)
async def ask_endpoint(payload: QueryRequest):
    """Async version with concurrency."""

    loop = asyncio.get_event_loop()

    ingest_task = loop.run_in_executor(None, ingest.main)
    pipeline_task = loop.run_in_executor(
        None, run_pipeline, payload.query, payload.top_k or 5
    )

    await ingest_task
    result = await pipeline_task

    return ChatResponse(
        answer=result.get("answer", "No answer found."),
        intent=result.get("intent"),
        sources=result.get("sources", []),
        formatted=True
    )
