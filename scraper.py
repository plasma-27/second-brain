import re
from typing import Optional, Dict
import requests
from bs4 import BeautifulSoup

URL_PATTERN = re.compile(r"https?://[^\s<>\"'{}|\\^`\[\]]+")


def extract_urls(text: str) -> list[str]:
    """Return all URLs found in the given text."""
    return URL_PATTERN.findall(text)


def scrape_url(url: str, max_chars: int = 12000) -> Dict[str, Optional[str]]:
    """Fetch a webpage and extract its title and cleaned body text."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    try:
        response = requests.get(url, headers=headers, timeout=12)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove non-content elements
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else url

        # Extract clean paragraphs and headings
        lines = [line.strip() for line in soup.get_text(separator="\n").splitlines()]
        clean_text = "\n".join(line for line in lines if len(line) > 20)

        return {
            "url": url,
            "title": title,
            "content": clean_text[:max_chars],
            "error": None,
        }
    except Exception as e:
        return {
            "url": url,
            "title": url,
            "content": "",
            "error": str(e),
        }