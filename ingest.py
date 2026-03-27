import os
import json
import sys
import subprocess
from typing import List, Dict

# Optional imports for document handling
try:
    from pypdf import PdfReader
except ImportError:
    print("🔧 Installing pypdf...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf"])
        from pypdf import PdfReader
    except Exception:
        PdfReader = None
        print("⚠️ pypdf not installed. PDF support disabled.")

try:
    from docx import Document
except ImportError:
    print("🔧 Installing python-docx...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "python-docx"])
        from docx import Document
    except Exception:
        Document = None
        print("⚠️ python-docx not installed. DOCX support disabled.")

from chunker import TextChunker
from vector_store import VectorStore


def load_web_data(file_path="raw_data.json") -> List[Dict]:
    """Loads crawled website data."""
    if not os.path.exists(file_path):
        print(f"⚠️ {file_path} not found. Only local documents will be used.")
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"🌐 Loaded {len(data)} pages from web crawl.")
    return data


def load_local_documents(folder_path="docs") -> List[Dict]:
    """Scans a folder for PDF, DOCX, and TXT files."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"📁 Created '{folder_path}' folder. Place your documents there.")
        return []

    docs = []
    print(f"📂 Scanning '{folder_path}' for documents...")

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        content = ""
        
        try:
            if filename.lower().endswith(".pdf") and PdfReader:
                reader = PdfReader(file_path)
                for page in reader.pages:
                    content += (page.extract_text() or "") + "\n"
            
            elif filename.lower().endswith(".docx") and Document:
                doc = Document(file_path)
                content = "\n".join([p.text for p in doc.paragraphs])
            
            elif filename.lower().endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            
            else:
                continue

            if content.strip():
                # Use filename as the "URL" for citations
                docs.append({"url": f"doc:{filename}", "content": content})
                print(f"   ✅ Loaded: {filename}")

        except Exception as e:
            print(f"   ❌ Error reading {filename}: {e}")

    return docs


def main():
    print("🚀 Starting ingestion pipeline...")

    # 1. Gather Data
    web_data = load_web_data()
    local_data = load_local_documents()

    if not web_data and not local_data:
        print("❌ No data found. Run crawler.py or add files to 'docs/' folder.")
        return

    chunker = TextChunker(chunk_size=500, overlap=50)
    
    # 2. Process & Chunk
    print("🔪 Chunking data...")
    
    # Process Web Data (is_web=True)
    web_chunks = chunker.create_chunks(web_data, is_web=True)
    
    # Process Doc Data (is_web=False)
    doc_chunks = chunker.create_chunks(local_data, is_web=False)
    
    final_chunks = web_chunks + doc_chunks
    print(f"🧩 Created {len(final_chunks)} total chunks.")

    # 3. Save Chunks (for Keyword Search)
    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(final_chunks, f, indent=2, ensure_ascii=False)

    # 4. Generate Embeddings (for Vector Search)
    vs = VectorStore()
    vs.create_embeddings_from_chunks(final_chunks, "embeddings.json")

    print("✨ Ingestion complete! Restart mcp_server.py to apply changes.")

if __name__ == "__main__":
    main()