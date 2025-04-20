import pytest
from src.query_processor import QueryProcessor
from src.config import Config

@pytest.fixture
def config():
    return Config()

@pytest.fixture
def query_processor(config):
    return QueryProcessor(config)

def test_query_processor_initialization(query_processor):
    """Test QueryProcessor initialization."""
    assert query_processor is not None
    assert hasattr(query_processor, 'rewrite')
    assert hasattr(query_processor, 'process_and_retrieve')

def test_rewrite_simple_query(query_processor):
    """Test rewriting a simple query."""
    query = "What is the capital of France?"
    result = query_processor.rewrite(query)
    assert result is not None
    assert 'processed_query' in result
    assert 'is_complex' in result
    assert result['is_complex'] is False

def test_rewrite_complex_query(query_processor):
    """Test rewriting a complex query."""
    query = "Compare and contrast the economic policies of France and Germany"
    result = query_processor.rewrite(query)
    assert result is not None
    assert 'processed_query' in result
    assert 'is_complex' in result
    assert result['is_complex'] is True
    assert 'sub_questions' in result
    assert len(result['sub_questions']) > 0

def test_process_and_retrieve(query_processor):
    """Test the complete process_and_retrieve workflow."""
    query = "What is the capital of France?"
    result = query_processor.process_and_retrieve(query, n_results=3)
    assert result is not None
    assert 'answer' in result
    assert 'sources' in result
    assert 'query_analysis' in result 