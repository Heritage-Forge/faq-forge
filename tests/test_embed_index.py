import pytest
from src.embed_index import EmbedIndexer

class TestEmbedIndexer:
    @pytest.fixture(scope="class")
    def indexer(self):
        return EmbedIndexer(model_name="all-MiniLM-L6-v2", index_path="test_index.faiss")

    def test_embedding_dimensions(self, indexer):
        texts = ["Test sentence one.", "Another test sentence."]
        embeddings = indexer.embed_texts(texts)
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] == indexer.dimension

    def test_build_index(self, indexer):
        texts = ["Test sentence one.", "Another test sentence.", "More text for testing."]
        embeddings = indexer.embed_texts(texts)
        index = indexer.build_index(embeddings)
        
        assert index.ntotal == len(texts)

    def test_save_and_load_index(self, indexer, tmp_path):
        texts = ["Test sentence one.", "Another test sentence."]
        embeddings = indexer.embed_texts(texts)
        index = indexer.build_index(embeddings)
        
        temp_index_file = tmp_path / "temp_index.faiss"
        indexer.save_index(index, str(temp_index_file))
        
        loaded_index = indexer.load_index(str(temp_index_file))
        
        assert loaded_index.ntotal == index.ntotal

    def test_query_index(self, indexer):
        texts = [
            "What is the capital of France?",
            "How to change a tire?",
            "Best practices for car maintenance."
        ]
        embeddings = indexer.embed_texts(texts)
        index = indexer.build_index(embeddings)

        query = "Capital city of France?"
        distances, indices = indexer.query_index(query, index, top_k=1)

        assert indices[0][0] == 0
