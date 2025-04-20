import pytest
from src.retriever import Retriever
from src.config import Config

@pytest.fixture
def config():
    return Config()

@pytest.fixture
def retriever(config):
    return Retriever(config)

def test_retriever_initialization(retriever):
    """Test Retriever initialization."""
    assert retriever is not None
    assert hasattr(retriever, 'retrieve')

def test_retrieve_documents(retriever):
    """Test document retrieval."""
    query = "What is the capital of France?"
    results = retriever.retrieve(query, n_results=3)
    assert results is not None
    assert isinstance(results, list)
    assert len(results) <= 3
    for result in results:
        assert 'text' in result
        assert 'metadata' in result
        assert 'score' in result

def test_retrieve_with_metadata(retriever):
    """Test document retrieval with metadata filtering."""
    query = "What is the capital of France?"
    metadata = {"source": "wikipedia"}
    results = retriever.retrieve(query, n_results=3, metadata=metadata)
    assert results is not None
    assert isinstance(results, list)
    for result in results:
        assert 'metadata' in result
        assert result['metadata'].get('source') == 'wikipedia'

def test_retrieve_empty_query(retriever):
    """Test document retrieval with empty query."""
    with pytest.raises(ValueError):
        retriever.retrieve("", n_results=3) 