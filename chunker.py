import json
import re


class TextChunker:
    def __init__(self, chunk_size=500, overlap=100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    # 🔥 STEP 1: CLEAN TEXT
    def clean_text(self, text, is_web=True):
        # Normalize spaces
        text = re.sub(r'\s+', ' ', text)

        # Remove common junk patterns
        junk_patterns = [
            r'©.*',
            r'All rights reserved.*',
            r'Privacy Policy.*',
            r'Terms & Conditions.*',
            r'Contact Us.*',
            r'Follow us.*',
            r'Login.*',
            r'Sign up.*',
            r'Cart.*',
        ]

        for pattern in junk_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # 🔥 REMOVE NAVIGATION WORDS (Only for Web)
        if is_web:
            nav_words = [
                "home", "about", "contact", "properties",
                "login", "signup", "menu"
            ]

            words = text.split()
            filtered_words = [w for w in words if w.lower() not in nav_words]

            text = " ".join(filtered_words)

        # 🔥 REMOVE SHORT LINES
        lines = re.split(r'[.!?]', text)
        lines = [line.strip() for line in lines if len(line.strip()) > 20]

        return ". ".join(lines)

    # 🔥 STEP 2: SPLIT INTO SENTENCES
    def split_sentences(self, text):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    # 🔥 STEP 3: CREATE SMART CHUNKS
    def create_smart_chunks(self, sentences):
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += " " + sentence
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    # 🔥 MAIN FUNCTION
    def create_chunks(self, data, is_web=True):
        all_chunks = []

        for page in data:
            url = page.get("url")
            content = page.get("content", "")

            if not content:
                continue

            # ✅ Clean text
            cleaned_text = self.clean_text(content, is_web=is_web)

            # ✅ Split into sentences
            sentences = self.split_sentences(cleaned_text)

            # ✅ Create chunks
            chunks = self.create_smart_chunks(sentences)

            for chunk in chunks:
                all_chunks.append({
                    "url": url,
                    "text": chunk
                })

        return all_chunks


# 🔥 RUN
if __name__ == "__main__":
    with open("raw_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    chunker = TextChunker(chunk_size=500, overlap=100)

    chunks = chunker.create_chunks(data)

    print(f"✅ Total Chunks Created: {len(chunks)}")

    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    # Preview
    for c in chunks[:3]:
        print("\nURL:", c["url"])
        print("Chunk Preview:", c["text"][:300])