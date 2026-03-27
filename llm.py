# llm.py
import os
import re
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CHOOSE YOUR LLM BACKEND
# Option A: Groq (FREE, FAST — Recommended)
# Option B: OpenAI
# Option C: Ollama (Local, FREE)
# Option D: Fallback (no API key needed, basic but better)
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")       # Get free key at console.groq.com
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Auto-detect which backend to use
def detect_backend():
    if GROQ_API_KEY:
        return "groq"
    elif OPENAI_API_KEY:
        return "openai"
    else:
        try:
            import requests
            r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
            if r.status_code == 200:
                return "ollama"
        except Exception:
            pass
    return "fallback"

BACKEND = detect_backend()
print(f"🧠 LLM Backend: {BACKEND.upper()}")


# ============================================================
# QUERY INTENT DETECTOR
# ============================================================
class QueryIntent:
    LIST_KEYWORDS = [
        "list", "show", "all", "available", "give me", "display",
        "what are", "how many", "enumerate", "properties", "apartments",
        "villas", "houses", "plots", "options", "types"
    ]
    PRICE_KEYWORDS = ["price", "cost", "rate", "fee", "charges", "how much", "budget"]
    LOCATION_KEYWORDS = ["where", "location", "address", "area", "city", "near"]
    CONTACT_KEYWORDS = ["contact", "phone", "email", "call", "reach", "office"]
    GREETING_KEYWORDS = ["hi", "hello", "hey", "good morning", "good evening"]

    @staticmethod
    def detect(query: str) -> str:
        q = query.lower()
        if any(k in q for k in QueryIntent.GREETING_KEYWORDS):
            return "greeting"
        if any(k in q for k in QueryIntent.LIST_KEYWORDS):
            return "list"
        if any(k in q for k in QueryIntent.PRICE_KEYWORDS):
            return "price"
        if any(k in q for k in QueryIntent.LOCATION_KEYWORDS):
            return "location"
        if any(k in q for k in QueryIntent.CONTACT_KEYWORDS):
            return "contact"
        return "general"


# ============================================================
# PROMPT BUILDER
# ============================================================
class PromptBuilder:
    SYSTEM_PROMPT = """You are an expert real estate assistant for a real estate company.
Your job is to answer user queries accurately based ONLY on the provided context.

Rules:
- Answer ONLY from the context provided. Do NOT make up information.
- If the context has property listings, format them as a numbered list.
- If asked for prices, give the exact price from context.
- If information is NOT in context, say: "I don't have that information right now. Please contact our team."
- Keep answers concise and helpful.
- For lists, use clear bullet points or numbering.
- Always be professional and friendly.
"""

    @staticmethod
    def build(query: str, context: str, intent: str, chunks: List[Dict]) -> str:
        # Build context with source info
        context_block = ""
        for i, chunk in enumerate(chunks):
            source = chunk.get("url", "website")
            source_label = source.replace("doc:", "📄 Document: ").replace("http://", "").replace("https://", "")
            context_block += f"\n[Source {i+1}: {source_label}]\n{chunk['text']}\n"

        # Intent-specific instructions
        intent_instruction = ""
        if intent == "list":
            intent_instruction = "\n⚠️ IMPORTANT: The user wants a LIST. Format your answer as a numbered or bulleted list. Extract ALL relevant items from context."
        elif intent == "price":
            intent_instruction = "\n⚠️ IMPORTANT: The user wants PRICE information. Extract and highlight prices clearly."
        elif intent == "location":
            intent_instruction = "\n⚠️ IMPORTANT: The user wants LOCATION/ADDRESS information. Be specific about locations."
        elif intent == "contact":
            intent_instruction = "\n⚠️ IMPORTANT: The user wants CONTACT details. Provide phone, email, or address if available."

        user_prompt = f"""Context from our real estate database:
{context_block}

{intent_instruction}

User Question: {query}

Answer (based only on the context above):"""

        return user_prompt


# ============================================================
# GROQ BACKEND (FREE & FAST — Recommended)
# ============================================================
class GroqLLM:
    def __init__(self):
        from groq import Groq
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = "llama-3.1-8b-instant"  # Free tier, very fast

    def generate(self, query: str, context: str, chunks: List[Dict], intent: str) -> str:
        prompt = PromptBuilder.build(query, context, intent, chunks)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": PromptBuilder.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,   # Low = more factual, less creative
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating answer: {e}"


# ============================================================
# OPENAI BACKEND
# ============================================================
class OpenAILLM:
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = "gpt-4o-mini"  # Cheap & accurate

    def generate(self, query: str, context: str, chunks: List[Dict], intent: str) -> str:
        prompt = PromptBuilder.build(query, context, intent, chunks)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": PromptBuilder.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {e}"


# ============================================================
# OLLAMA BACKEND (Local, Free)
# ============================================================
class OllamaLLM:
    def __init__(self):
        import requests
        self.base_url = OLLAMA_BASE_URL
        # Auto-detect best available model
        try:
            r = requests.get(f"{self.base_url}/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
            preferred = ["llama3.1", "llama3", "mistral", "llama2"]
            self.model = next((m for p in preferred for m in models if p in m), models[0] if models else "llama3")
            print(f"   Using Ollama model: {self.model}")
        except Exception:
            self.model = "llama3"

    def generate(self, query: str, context: str, chunks: List[Dict], intent: str) -> str:
        import requests
        prompt = PromptBuilder.build(query, context, intent, chunks)
        full_prompt = f"{PromptBuilder.SYSTEM_PROMPT}\n\n{prompt}"
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": full_prompt, "stream": False},
                timeout=120
            )
            return response.json().get("response", "").strip()
        except Exception as e:
            return f"Error: {e}"


# ============================================================
# FALLBACK (No API — Smarter Extraction)
# ============================================================
class FallbackLLM:
    """
    Smarter fallback when no LLM API is available.
    Uses intent + keyword matching to build better answers.
    """
    def generate(self, query: str, context: str, chunks: List[Dict], intent: str) -> str:
        if not context:
            return "I don't have that information right now. Please contact our team."

        sentences = re.split(r'(?<=[.!?])\s+', context)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 30]

        query_words = set(query.lower().split())

        # Score sentences by relevance to query
        scored = []
        for s in sentences:
            s_lower = s.lower()
            score = sum(1 for word in query_words if word in s_lower)
            scored.append((score, s))

        scored.sort(reverse=True)

        if intent == "list":
            # Try to extract list items
            top = [s for score, s in scored if score > 0][:8]
            if top:
                return "Here are the relevant results:\n" + "\n".join(f"• {s}" for s in top)
            return "I couldn't find a specific list. Please check our properties page."

        elif intent == "price":
            # Prioritize sentences with price-related numbers
            price_sents = [s for score, s in scored if any(
                c.isdigit() for c in s
            )][:3]
            if price_sents:
                return " ".join(price_sents)

        # General: return top 3 most relevant sentences
        top = [s for score, s in scored if score > 0][:3]
        if not top:
            top = [s for score, s in scored[:3]]  # fallback: just top 3

        return " ".join(top) if top else "I don't have that information right now."


# ============================================================
# MAIN LLM CLASS (Auto-selects backend)
# ============================================================
class LLM:
    def __init__(self):
        self.intent_detector = QueryIntent()
        if BACKEND == "groq":
            self._engine = GroqLLM()
        elif BACKEND == "openai":
            self._engine = OpenAILLM()
        elif BACKEND == "ollama":
            self._engine = OllamaLLM()
        else:
            print("⚠️ No LLM API found. Using smart fallback. Set GROQ_API_KEY for best results.")
            self._engine = FallbackLLM()

    def generate(self, query: str, context: str, chunks: List[Dict] = None) -> str:
        intent = QueryIntent.detect(query)
        chunks = chunks or []
        return self._engine.generate(query, context, chunks, intent)
