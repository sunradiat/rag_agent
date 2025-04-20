import pytest
from src.reranker import Reranker
from src.config import Config

@pytest.fixture
def config():
    return Config()

@pytest.fixture
def reranker(config):
    return Reranker(config)

@pytest.fixture
def sample_documents():
    return [
        {
            "text": "Paris is the capital of France.",
            "metadata": {"source": "wikipedia"},
            "score": 0.9
        },
        {
            "text": "France is a country in Europe.",
            "metadata": {"source": "encyclopedia"},
            "score": 0.8
        },
        {
            "text": "The Eiffel Tower is in Paris.",
            "metadata": {"source": "travel_guide"},
            "score": 0.7
        }
    ]

def test_reranker_initialization(reranker):
    """Test Reranker initialization."""
    assert reranker is not None
    assert hasattr(reranker, 'rerank')

def test_rerank_documents(reranker, sample_documents):
    """Test document reranking."""
    query = "What is the capital of France?"
    reranked = reranker.rerank(query, sample_documents)
    assert reranked is not None
    assert isinstance(reranked, list)
    assert len(reranked) == len(sample_documents)
    for doc in reranked:
        assert 'text' in doc
        assert 'metadata' in doc
        assert 'score' in doc
        assert isinstance(doc['score'], float)

def test_rerank_empty_documents(reranker):
    """Test reranking with empty document list."""
    query = "What is the capital of France?"
    reranked = reranker.rerank(query, [])
    assert reranked is not None
    assert isinstance(reranked, list)
    assert len(reranked) == 0

def test_rerank_invalid_documents(reranker):
    """Test reranking with invalid document format."""
    query = "What is the capital of France?"
    invalid_docs = [{"text": "Paris"}, {"text": "London"}]
    with pytest.raises(ValueError):
        reranker.rerank(query, invalid_docs) 