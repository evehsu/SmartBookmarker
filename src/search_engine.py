import numpy as np
from typing import List, Tuple
import json

class SearchEngine:
    def __init__(self, db_handler, embedding_service):
        self.db_handler = db_handler
        self.embedding_service = embedding_service

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """Search bookmarks based on query similarity"""
        # Get query embedding
        query_embedding = self.embedding_service.get_embedding(query)
        
        if not query_embedding:
            print("query embedding is not available")
            return []

        # Get all bookmarks
        bookmarks = self.db_handler.get_all_bookmarks()
        
        if not bookmarks:
            print("bookmarks are not available in local database")
            return []

        # Calculate similarities
        similarities = []
        for url, name, embedding_blob in bookmarks:
            embedding = np.array(json.loads(embedding_blob))
            similarity = self._calculate_similarity(query_embedding, embedding)
            similarities.append((url, name, similarity))

        # Sort by similarity and return top k results
        similarities.sort(key=lambda x: x[2], reverse=True)
        return similarities[:top_k]

    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)) 