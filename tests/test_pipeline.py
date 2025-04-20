import pytest
from src.pipeline import DataIngestionPipeline, QueryPipeline
from src.config import Config

@pytest.fixture
def config():
    return Config()

@pytest.fixture
def ingestion_pipeline(config):
    return DataIngestionPipeline(config)

@pytest.fixture
def query_pipeline(config):
    return QueryPipeline(config)

def test_data_ingestion_pipeline_initialization(ingestion_pipeline):
    """Test DataIngestionPipeline initialization."""
    assert ingestion_pipeline is not None
    assert hasattr(ingestion_pipeline, 'process_document')
    assert hasattr(ingestion_pipeline, 'process_folder')

def test_query_pipeline_initialization(query_pipeline):
    """Test QueryPipeline initialization."""
    assert query_pipeline is not None
    assert hasattr(query_pipeline, 'process_query')

def test_process_query_simple(query_pipeline):
    """Test processing a simple query."""
    result = query_pipeline.process_query(
        query="What is the capital of France?",
        n_results=3
    )
    assert result is not None
    assert 'answer' in result
    assert 'sources' in result
    assert 'query_analysis' in result

def test_process_query_complex(query_pipeline):
    """Test processing a complex query."""
    result = query_pipeline.process_query(
        query="Compare and contrast the economic policies of France and Germany",
        n_results=5
    )
    assert result is not None
    assert 'answer' in result
    assert 'sources' in result
    assert 'query_analysis' in result
    assert result['query_analysis']['is_complex'] is True 