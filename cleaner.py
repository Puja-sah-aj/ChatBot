import json
import re


class TextCleaner:
    def __init__(self):
        # Navigation/junk words (be more conservative)
        self.nav_phrases = [
            "skip to content",
            "skip to main",
            "back to top",
            "cookie policy",
            "accept cookies",
            "we use cookies"
        ]
        
        # Junk patterns (more specific to avoid removing valid content)
        self.junk_patterns = [
            r'©\s*\d{4}.*?(?=\.|$)',
            r'All rights reserved\.?',
            r'Privacy Policy\s*\|?\s*Terms',
            r'Follow us on \w+',
            r'Subscribe to our newsletter',
            r'Loading\.\.\.',
            r'Please wait\.\.\.'
        ]

    def clean_text(self, text):
        """Clean text while preserving meaningful content."""
        if not text:
            return ""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove junk patterns
        for pattern in self.junk_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove nav phrases
        for phrase in self.nav_phrases:
            text = re.sub(re.escape(phrase), '', text, flags=re.IGNORECASE)
        
        # Remove repeated words (like: Home Home Home)
        text = re.sub(r'\b(\w+)( \1\b){2,}', r'\1', text, flags=re.IGNORECASE)
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def clean_data(self, data):
        """Clean all crawled data."""
        cleaned_data = []
        
        for page in data:
            url = page.get("url", "")
            content = page.get("content", "")
            
            if not content:
                continue
            
            cleaned_text = self.clean_text(content)
            
            # Keep pages with meaningful content (lower threshold)
            if cleaned_text and len(cleaned_text) > 30:
                cleaned_data.append({
                    "url": url,
                    "content": cleaned_text
                })
        
        return cleaned_data


if __name__ == "__main__":
    with open("raw_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    cleaner = TextCleaner()
    cleaned = cleaner.clean_data(data)
    
    print(f"✅ Cleaned Pages: {len(cleaned)}")
    
    with open("cleaned_data.json", "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)
    
    # Preview
    for c in cleaned[:3]:
        print("\n" + "="*50)
        print("URL:", c["url"])
        print("Content Preview:", c["content"][:500])
