import pytest
from src.query_decomposer import QueryDecomposer
from src.config import Config

@pytest.fixture
def config():
    return Config()

@pytest.fixture
def query_decomposer(config):
    return QueryDecomposer(config)

def test_query_decomposer_initialization(query_decomposer):
    """Test QueryDecomposer initialization."""
    assert query_decomposer is not None
    assert hasattr(query_decomposer, 'analyze_complexity')
    assert hasattr(query_decomposer, 'decompose')

def test_analyze_complexity_simple(query_decomposer):
    """Test complexity analysis for a simple query."""
    query = "What is the capital of France?"
    result = query_decomposer.analyze_complexity(query)
    assert result is not None
    assert 'is_complex' in result
    assert 'explanation' in result
    assert result['is_complex'] is False

def test_analyze_complexity_complex(query_decomposer):
    """Test complexity analysis for a complex query."""
    query = "Compare and contrast the economic policies of France and Germany"
    result = query_decomposer.analyze_complexity(query)
    assert result is not None
    assert 'is_complex' in result
    assert 'explanation' in result
    assert result['is_complex'] is True

def test_decompose_query(query_decomposer):
    """Test query decomposition."""
    query = "Compare and contrast the economic policies of France and Germany"
    result = query_decomposer.decompose(query)
    assert result is not None
    assert 'sub_questions' in result
    assert len(result['sub_questions']) > 0
    assert all(isinstance(q, str) for q in result['sub_questions'])
    assert 'dependencies' in result
    assert isinstance(result['dependencies'], list) 