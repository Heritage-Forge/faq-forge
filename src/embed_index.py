import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbedIndexer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", index_path: str = "data/faq.index"):
        """
        Initialize the EmbedIndexer with a SentenceTransformer model.
        
        Args:
            model_name (str): Name of the embedding model.
            index_path (str): Default file path to save/load the FAISS index.
        """
        self.model_name = model_name
        self.index_path = index_path
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts (list[str]): List of strings to embed.
            
        Returns:
            np.ndarray: Embeddings array with shape (num_texts, dimension).
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        embeddings = embeddings.astype("float32")
        # Normalize embeddings for cosine similarity using inner product.
        faiss.normalize_L2(embeddings)
        return embeddings

    def build_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Build a FAISS index from the embeddings.
        
        Args:
            embeddings (np.ndarray): Array of embeddings.
            
        Returns:
            faiss.Index: A FAISS index with added embeddings.
        """
        index = faiss.IndexFlatIP(self.dimension)
        index.add(embeddings)
        return index

    def save_index(self, index: faiss.Index, index_path: str = None) -> None:
        """
        Save the FAISS index to disk.
        
        Args:
            index (faiss.Index): The FAISS index to save.
            index_path (str, optional): Path to save the index. Defaults to self.index_path.
        """
        if index_path is None:
            index_path = self.index_path
        faiss.write_index(index, index_path)

    def load_index(self, index_path: str = None) -> faiss.Index:
        """
        Load a FAISS index from disk.
        
        Args:
            index_path (str, optional): Path to the index file. Defaults to self.index_path.
            
        Returns:
            faiss.Index: The loaded FAISS index.
        """
        if index_path is None:
            index_path = self.index_path
        return faiss.read_index(index_path)

    def query_index(self, query_text: str, index: faiss.Index, top_k: int = 3) -> tuple[np.ndarray, np.ndarray]:
        """
        Query the FAISS index with a text query.
        
        Args:
            query_text (str): The text query.
            index (faiss.Index): A loaded FAISS index.
            top_k (int): Number of nearest neighbors to retrieve.
            
        Returns:
            tuple: (distances, indices) from the index search, with distances clamped to [-1, 1].
        """
        # Ensure we don't request more neighbors than available
        effective_top_k = min(top_k, index.ntotal)
        
        query_embedding = self.embed_texts([query_text])
        distances, indices = index.search(query_embedding, effective_top_k)
        # Clamp distances to the range [-1, 1] to avoid extreme negative values
        distances = np.clip(distances, -1.0, 1.0)
        return distances, indices
