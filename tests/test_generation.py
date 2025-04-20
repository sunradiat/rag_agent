import pytest
from src.generation import Generator
from src.config import Config

@pytest.fixture
def config():
    return Config()

@pytest.fixture
def generator(config):
    return Generator(config)

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
        }
    ]

def test_generator_initialization(generator):
    """Test Generator initialization."""
    assert generator is not None
    assert hasattr(generator, 'generate')
    assert hasattr(generator, 'combine_answers')

def test_generate_answer(generator, sample_documents):
    """Test answer generation."""
    query = "What is the capital of France?"
    result = generator.generate(query, sample_documents)
    assert result is not None
    assert 'answer' in result
    assert 'sources' in result
    assert isinstance(result['answer'], str)
    assert len(result['answer']) > 0
    assert isinstance(result['sources'], list)
    assert len(result['sources']) > 0

def test_combine_answers(generator):
    """Test combining multiple answers."""
    original_query = "Compare and contrast the economic policies of France and Germany"
    sub_answers = [
        {
            "answer": "France has a mixed economy with strong government intervention.",
            "sources": ["source1"]
        },
        {
            "answer": "Germany follows a social market economy model.",
            "sources": ["source2"]
        }
    ]
    result = generator.combine_answers(original_query, sub_answers)
    assert result is not None
    assert 'answer' in result
    assert 'sources' in result
    assert isinstance(result['answer'], str)
    assert len(result['answer']) > 0
    assert isinstance(result['sources'], list)
    assert len(result['sources']) > 0

def test_generate_empty_documents(generator):
    """Test answer generation with empty documents."""
    query = "What is the capital of France?"
    with pytest.raises(ValueError):
        generator.generate(query, []) 