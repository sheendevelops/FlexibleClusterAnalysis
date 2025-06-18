import trafilatura
import requests
from typing import Optional

def get_website_text_content(url: str) -> Optional[str]:
    """
    This function takes a url and returns the main text content of the website.
    The text content is extracted using trafilatura and easier to understand.
    The results is not directly readable, better to be summarized by LLM before consume
    by the user.

    Some common website to crawl information from:
    MLB scores: https://www.mlb.com/scores/YYYY-MM-DD
    """
    try:
        # Send a request to the website
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            return None
        
        # Extract text content
        text = trafilatura.extract(downloaded)
        return text
    
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return None

def get_website_with_metadata(url: str) -> dict:
    """
    Get website content along with metadata
    """
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            return {"error": "Could not fetch URL"}
        
        text = trafilatura.extract(downloaded)
        metadata = trafilatura.extract_metadata(downloaded)
        
        return {
            "content": text,
            "title": metadata.title if metadata else "Unknown",
            "author": metadata.author if metadata else "Unknown",
            "date": metadata.date if metadata else "Unknown",
            "url": url
        }
    
    except Exception as e:
        return {"error": f"Error processing {url}: {str(e)}"}

def is_valid_url(url: str) -> bool:
    """
    Check if URL is valid and accessible
    """
    try:
        response = requests.head(url, timeout=10)
        return response.status_code == 200
    except Exception:
        return False

def extract_links_from_content(content: str) -> list:
    """
    Extract potential URLs from text content
    """
    import re
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(url_pattern, content)
