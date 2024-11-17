import sqlite3
import json
import numpy as np

class DatabaseHandler:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """Initialize the database with necessary tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bookmarks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT NOT NULL,
                    name TEXT,
                    embedding BLOB,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bookmark_hash (
                    url TEXT PRIMARY KEY,
                    content_hash TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def store_bookmark(self, url: str, name: str, embedding: list):
        """Store a bookmark with its embedding"""
        embedding_blob = json.dumps(embedding)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO bookmarks (url, name, embedding)
                VALUES (?, ?, ?)
            """, (url, name, embedding_blob))

    def store_bookmark_hash(self, url: str, content_hash: str):
        """Store a bookmark's content hash"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO bookmark_hash (url, content_hash)
                VALUES (?, ?)
            """, (url, content_hash))

    def get_bookmark_hash(self, url: str) -> str:
        """Get a bookmark's content hash"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT content_hash FROM bookmark_hash WHERE url = ?
            """, (url,))
            result = cursor.fetchone()
            return result[0] if result else None

    def get_all_bookmarks(self):
        """Get all bookmarks"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT url, name, embedding FROM bookmarks
            """)
            return cursor.fetchall()

    def delete_bookmark(self, url: str):
        """Delete a bookmark"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM bookmarks WHERE url = ?", (url,))
            conn.execute("DELETE FROM bookmark_hash WHERE url = ?", (url,)) 