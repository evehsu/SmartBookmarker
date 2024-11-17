import openai
import os
from dotenv import load_dotenv
import numpy as np
from openai import OpenAI


class EmbeddingService:
    def __init__(self):
        load_dotenv()
        openai.api_key = os.getenv('OPENAI_API_KEY')
        self.model = "text-embedding-ada-002"
        self.client = OpenAI()

    def get_embedding(self, text: str) -> list:
        """Get embedding for a single text using OpenAI API"""
        embedding = []
        try:
            embedding = self.client.embeddings.create(input = [text], model="text-embedding-ada-002").data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
        return embedding

    def get_embeddings_batch(self, texts: list[str], batch_size: int = 100) -> list:
        """Get embeddings for multiple texts in batches"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = openai.Embedding.create(
                    input=batch,
                    model=self.model
                )
                embeddings = [data['embedding'] for data in response['data']]
                all_embeddings.extend(embeddings)
            except Exception as e:
                print(f"Error getting embeddings for batch: {e}")
                return None
                
        return all_embeddings 