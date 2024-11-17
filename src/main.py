from embedding_service import EmbeddingService
from db_handler import DatabaseHandler
from bookmark_processor import BookmarkProcessor
from search_engine import SearchEngine
import json
import os

class BookmarkSearchApp:
    def __init__(self, db_path: str):
        self.db_handler = DatabaseHandler(db_path)
        self.embedding_service = EmbeddingService()
        self.bookmark_processor = BookmarkProcessor(self.db_handler, self.embedding_service)
        self.search_engine = SearchEngine(self.db_handler, self.embedding_service)

    def update_bookmarks(self, bookmarks_path: str):
        """Update bookmarks with new data"""
        with open(bookmarks_path, 'r') as f:
            bookmarks_data = json.load(f)
        bookmarks = self.bookmark_processor.process_chrome_bookmarks(bookmarks_data)
        self.bookmark_processor.process_bookmarks(bookmarks)

    def search(self, query: str, top_k: int = 5):
        """Search bookmarks"""
        results = self.search_engine.search(query, top_k)
        return results

def main():
    # Initialize the application
    app = BookmarkSearchApp('bookmarks.db')

    # Example usage
    USER = "evelyn_xu"
    bookmarks_path = f"/Users/{USER}/Library/application support/Google/Chrome/Profile 1/Bookmarks"
    

    app.update_bookmarks(bookmarks_path)

    # Example search
    query = input("Enter your search query: ")
    results = app.search(query)
    print(results)
    print("\nSearch Results:")
    for url, title, similarity in results:
        print(f"\nTitle: {title}")
        print(f"URL: {url}")
        print(f"Similarity: {similarity:.4f}")

if __name__ == "__main__":
    main() 