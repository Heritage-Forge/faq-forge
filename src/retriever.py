import json
from pathlib import Path
from src.embed_index import EmbedIndexer

class Retriever:
    def __init__(self, index_path: str, data_path: str, model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the retriever with a saved FAISS index and corresponding FAQ data.
        
        Args:
            index_path (str): Path to the saved FAISS index.
            data_path (str): Path to the cleaned FAQ JSON data.
            model (str): Embedding model to use.
        """
        self.index_path = index_path
        self.data_path = data_path
        self.embed_indexer = EmbedIndexer(model_name=model, index_path=index_path)
        self.index = self.embed_indexer.load_index()
        self.faq_data = self._load_data()

    def _load_data(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def retrieve(self, query: str, top_k: int = 3):
        """
        Retrieve top-k FAQ items given a text query.
        
        Args:
            query (str): User query text.
            top_k (int): Number of results to return.
            
        Returns:
            list: List of dictionaries with keys "score", "question", "answer".
        """
        distances, indices = self.embed_indexer.query_index(query, self.index, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.faq_data):
                item = self.faq_data[idx]
                results.append({
                    "score": float(dist),
                    "question": item.get("question"),
                    "answer": item.get("answer")
                })
        return results
