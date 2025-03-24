import json
import pytest
from pathlib import Path
from src.embed_index import EmbedIndexer
from src.retriever import Retriever

@pytest.fixture
def sample_faq(tmp_path: Path) -> Path:
    data = [
        {
            "question": "Do you sell 3D-printed parts for 1967 Ford Mustangs?",
            "answer": "Yes, we offer precision-engineered 3D printed parts for 1965–1970 Mustangs."
        },
        {
            "question": "What’s the expected delivery time for custom suspension components?",
            "answer": "Lead time is typically 2–4 weeks depending on production load."
        }
    ]
    file_path = tmp_path / "faq.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return file_path

@pytest.fixture
def faq_index_file(tmp_path: Path, sample_faq: Path) -> Path:
    with open(sample_faq, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = [item["question"] for item in data]
    
    embed_indexer = EmbedIndexer(model_name="all-MiniLM-L6-v2", index_path=str(tmp_path / "faq.index"))
    embeddings = embed_indexer.embed_texts(questions)
    index = embed_indexer.build_index(embeddings)
    
    index_file = tmp_path / "faq.index"
    embed_indexer.save_index(index, str(index_file))
    return index_file

def test_retriever(sample_faq: Path, faq_index_file: Path):
    retriever = Retriever(
        index_path=str(faq_index_file),
        data_path=str(sample_faq),
        model="all-MiniLM-L6-v2"
    )
    query = "Do you sell parts for mustang?"
    results = retriever.retrieve(query, top_k=2)
    
    assert len(results) > 0, "Expected at least one retrieved result."
    
    mustang_found = any("mustangs" in res["question"].lower() for res in results)
    assert mustang_found, "Mustang-related FAQ not found in results."
    
    for res in results:
        assert -1.0 <= res["score"] <= 1.0, "Score out of range."
