import asyncio
from playwright.async_api import async_playwright
from urllib.parse import urljoin, urlparse, urlunparse
from bs4 import BeautifulSoup
import json
import re
import os
from dotenv import load_dotenv

load_dotenv()


class DynamicCrawler:
    def __init__(self, base_url, max_pages=100):
        self.base_url = base_url.rstrip('/')
        self.domain = urlparse(base_url).netloc
        self.scheme = urlparse(base_url).scheme
        
        self.visited = set()
        self.to_visit = [self.base_url]
        self.max_pages = max_pages
        
        self.results = []
        
        # Common file extensions to skip
        self.skip_extensions = {
            '.pdf', '.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp',
            '.mp4', '.mp3', '.wav', '.avi', '.mov',
            '.zip', '.rar', '.tar', '.gz',
            '.css', '.js', '.json', '.xml',
            '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx'
        }

    def normalize_url(self, url):
        """Normalize URL to avoid duplicates."""
        parsed = urlparse(url)
        
        # Remove fragments and query params for comparison
        normalized = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path.rstrip('/') or '/',
            '',  # params
            '',  # query (remove for deduplication, or keep if needed)
            ''   # fragment
        ))
        return normalized

    def should_skip_url(self, url):
        """Check if URL should be skipped."""
        parsed = urlparse(url)
        path = parsed.path.lower()
        
        # Skip files with certain extensions
        for ext in self.skip_extensions:
            if path.endswith(ext):
                return True
        
        # Skip common non-content paths
        skip_patterns = [
            '/api/', '/static/', '/assets/', '/images/', '/img/',
            '/cdn/', '/_next/', '/__', '/node_modules/'
        ]
        for pattern in skip_patterns:
            if pattern in path:
                return True
        
        return False

    async def fetch_page(self, page, url):
        """Fetch page content with multiple retry strategies."""
        try:
            print(f"🔍 Crawling: {url}")
            
            # First attempt: wait for network idle
            response = await page.goto(
                url,
                timeout=30000,
                wait_until="networkidle"
            )
            
            if response and response.status >= 400:
                print(f"⚠️ HTTP {response.status}: {url}")
                return None
            
            # Wait for dynamic content to load
            await page.wait_for_timeout(2000)
            
            # Try to wait for main content
            try:
                await page.wait_for_selector('main, article, .content, #content, [role="main"]', timeout=5000)
            except:
                pass  # Continue even if no main content selector found
            
            # Scroll to trigger lazy loading
            await self._scroll_page(page)
            
            content = await page.content()
            return content

        except Exception as e:
            print(f"⚠️ First attempt failed for {url}: {e}")
            
            # Retry with less strict waiting
            try:
                await page.goto(url, timeout=30000, wait_until="domcontentloaded")
                await page.wait_for_timeout(3000)
                return await page.content()
            except Exception as e2:
                print(f"❌ Error crawling {url}: {e2}")
                return None

    async def _scroll_page(self, page):
        """Scroll page to trigger lazy loading."""
        try:
            await page.evaluate('''
                async () => {
                    const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
                    const height = document.body.scrollHeight;
                    const step = window.innerHeight;
                    
                    for (let i = 0; i < height; i += step) {
                        window.scrollTo(0, i);
                        await delay(100);
                    }
                    window.scrollTo(0, 0);
                }
            ''')
            await page.wait_for_timeout(500)
        except:
            pass

    def extract_links(self, html, current_url):
        """Extract all internal links from the page."""
        soup = BeautifulSoup(html, "html.parser")
        links = set()

        # Find all anchor tags
        for tag in soup.find_all("a", href=True):
            href = tag["href"].strip()
            
            # Skip empty, javascript, and mailto links
            if not href or href.startswith(('javascript:', 'mailto:', 'tel:', '#')):
                continue
            
            # Convert to absolute URL
            full_url = urljoin(current_url, href)
            parsed = urlparse(full_url)
            
            # Only follow links on same domain
            if parsed.netloc != self.domain:
                continue
            
            # Normalize the URL
            normalized = self.normalize_url(full_url)
            
            # Skip unwanted URLs
            if not self.should_skip_url(normalized):
                links.add(normalized)

        # Also look for links in onclick handlers and data attributes (for SPAs)
        for tag in soup.find_all(attrs={"onclick": True}):
            onclick = tag.get("onclick", "")
            urls = re.findall(r"['\"]([^'\"]*)['\"]", onclick)
            for url in urls:
                if url.startswith('/'):
                    full_url = urljoin(current_url, url)
                    normalized = self.normalize_url(full_url)
                    if not self.should_skip_url(normalized):
                        links.add(normalized)

        # Check for Next.js/React router links
        for tag in soup.find_all(attrs={"href": True}):
            href = tag.get("href", "")
            if href.startswith('/'):
                full_url = urljoin(current_url, href)
                normalized = self.normalize_url(full_url)
                if not self.should_skip_url(normalized):
                    links.add(normalized)

        return links

    def extract_text(self, html, url):
        """Extract meaningful text content from the page."""
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove unwanted tags
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "iframe"]):
            tag.decompose()
        
        # Try to find main content area
        main_content = None
        content_selectors = [
            'main',
            'article',
            '[role="main"]',
            '.content',
            '#content',
            '.main-content',
            '#main-content',
            '.page-content',
            '.post-content'
        ]
        
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        # If no main content found, use body
        if not main_content:
            main_content = soup.body if soup.body else soup
        
        # Extract text
        text = main_content.get_text(separator=" ", strip=True)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Also extract page title
        title = ""
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text(strip=True)
        
        # Extract meta description
        meta_desc = ""
        meta_tag = soup.find('meta', attrs={'name': 'description'})
        if meta_tag:
            meta_desc = meta_tag.get('content', '')
        
        # Combine title, meta description, and content
        full_text = f"{title}. {meta_desc} {text}".strip()
        
        return full_text

    async def crawl(self):
        """Main crawling loop."""
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-web-security",
                    "--disable-features=IsolateOrigins,site-per-process"
                ]
            )
            
            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            
            page = await context.new_page()
            
            # Enable JavaScript console logging for debugging
            page.on("console", lambda msg: None)  # Suppress console messages
            
            while self.to_visit and len(self.visited) < self.max_pages:
                url = self.to_visit.pop(0)
                normalized_url = self.normalize_url(url)
                
                if normalized_url in self.visited:
                    continue
                
                html = await self.fetch_page(page, url)
                
                if not html:
                    continue
                
                text = self.extract_text(html, url)
                
                # Only save if we got meaningful content
                if text and len(text) > 50:
                    self.visited.add(normalized_url)
                    self.results.append({
                        "url": url,
                        "content": text
                    })
                    print(f"✅ Saved: {url} ({len(text)} chars)")
                else:
                    print(f"⚠️ Skipped (no content): {url}")
                    self.visited.add(normalized_url)
                
                # Extract and queue new links
                links = self.extract_links(html, url)
                new_links = 0
                for link in links:
                    normalized_link = self.normalize_url(link)
                    if normalized_link not in self.visited and link not in self.to_visit:
                        self.to_visit.append(link)
                        new_links += 1
                
                print(f"📊 Visited: {len(self.visited)} | Queue: {len(self.to_visit)} | Found {new_links} new links")
            
            await browser.close()
        
        print(f"\n✅ Crawling complete! Total pages: {len(self.results)}")
        return self.results


def crawl(base_url=os.getenv("BASE_URL", "https://venerable-gelato-283258.netlify.app/"), max_pages=100):
    """Entry point for crawling."""
    crawler = DynamicCrawler(base_url, max_pages=max_pages)
    data = asyncio.run(crawler.crawl())
    
    # Save to file
    with open("raw_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved {len(data)} pages to raw_data.json")
    return data


if __name__ == "__main__":
    crawl()
