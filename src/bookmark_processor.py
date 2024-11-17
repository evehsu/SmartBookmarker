import hashlib
from typing import Dict, List, Tuple
import json
from bs4 import BeautifulSoup
import httpx
from transformers import GPT2Tokenizer

class BookmarkProcessor:
    def __init__(self, db_handler, embedding_service):
        self.db_handler = db_handler
        self.embedding_service = embedding_service
    
    def _scrape_url(self, url):
        try:
            with httpx.Client() as client:
                response = client.get(url)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                
                title = soup.title.string if soup.title else ""
                content = ' '.join([p.get_text(strip=True) for p in soup.find_all('p')])
                
                return {
                    "title": title,
                    "content": content
                }
        except httpx.RequestError as e:
            return {"error": f"An error occurred while requesting {url}: {str(e)}"}
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP error occurred: {e.response.status_code} {e.response.reason_phrase}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred: {str(e)}"}

    def _generate_bookmark_text(self, bookmark_item, max_length=7500):
        text = [bookmark_item['url'], bookmark_item['name'], bookmark_item['page_path']]
        # name is by default to be title, but if the name is user defined, we will append title as well
        if 'title' in bookmark_item and bookmark_item['name'] != bookmark_item['title']:
            text.append(bookmark_item['title'])
        if 'content' in bookmark_item:
            text.append(bookmark_item['content'])
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokens = tokenizer.encode(';'.join(text), max_length = max_length, truncation=True, return_tensors="pt")
        truncated_text = tokenizer.decode(tokens[0])
        return truncated_text
    
    def _extract_bookmarks(self, data, current_path=""):
        bookmarks = []
        if isinstance(data, dict):
            if data.get("type") == "url":
                bookmark_info = {
                    "url": data["url"],
                    "name": data["name"],
                    "page_path": current_path.strip("/")
                }
                bookmarks.append(bookmark_info)
            elif data.get("type") == "folder":
                new_path = f"{current_path}/{data['name']}"
                if "children" in data:
                    bookmarks.extend(self._extract_bookmarks(data["children"], new_path))
            elif "children" in data:
                bookmarks.extend(self._extract_bookmarks(data["children"], current_path))
        elif isinstance(data, list):
            for item in data:
                bookmarks.extend(self._extract_bookmarks(item, current_path))
        
        return bookmarks

    def process_chrome_bookmarks(self, bookmark_json):
        all_bookmarks = []
        
        for root_folder, content in bookmark_json['roots'].items():
            all_bookmarks.extend(self._extract_bookmarks(content, content["name"]))
        
        return all_bookmarks

    def process_bookmarks(self, bookmarks_data: List[Dict]) -> None:
        """Process new or updated bookmarks"""
        for bookmark in bookmarks_data:
            url = bookmark['url']
            bookmark.update(self._scrape_url(url))
            hash_text = self._generate_bookmark_text(bookmark)
            name = bookmark['name']
            
            # Create content hash
            content_hash = self._create_content_hash(hash_text)
            existing_hash = self.db_handler.get_bookmark_hash(url)

            # Only process if content has changed
            if existing_hash != content_hash:
                # Get embedding for the bookmark
                embedding = self.embedding_service.get_embedding(hash_text)
                if embedding:
                    # Store bookmark and its hash
                    self.db_handler.store_bookmark(url, name, embedding)
                    self.db_handler.store_bookmark_hash(url, content_hash)

    def _create_content_hash(self, hash_text:str) -> str:
        """Create a hash of the bookmark content"""
        content = f"{hash_text}".encode('utf-8')
        return hashlib.md5(content).hexdigest() 