# RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline for document processing and query answering.

## Project Structure

```
rag/
├── src/                    # Source code
│   ├── __init__.py        # Package initialization
│   ├── pipeline.py        # Main pipeline classes
│   ├── query_processor.py # Query processing
│   ├── query_decomposer.py# Query decomposition
│   ├── retriever.py       # Document retrieval
│   ├── reranker.py        # Result reranking
│   ├── generation.py      # Answer generation
│   ├── database_operation.py # Database operations
│   ├── chunking.py        # Text chunking
│   ├── embedding.py       # Embedding generation
│   ├── config.py          # Configuration settings
│   └── doc_processing.py  # Document processing utilities
├── tests/                 # Unit tests
│   ├── __init__.py       # Test package initialization
│   ├── test_pipeline.py  # Pipeline tests
│   ├── test_query_processor.py # Query processor tests
│   ├── test_query_decomposer.py # Query decomposer tests
│   ├── test_retriever.py # Retriever tests
│   ├── test_reranker.py  # Reranker tests
│   └── test_generation.py # Generator tests
├── test.py                # Test suite
├── rag_pipeline_demo.ipynb# Demo notebook
├── requirements.txt       # Dependencies
├── pyproject.toml        # Project configuration
├── .pre-commit-config.yaml # Pre-commit hooks
└── .gitignore            # Git ignore rules
```

## Installation

There are two ways to install this project:

### Method 1: Using requirements.txt

1. Create a virtual environment:
```bash
# Create a new virtual environment
python -m venv venv
```

2. Activate the virtual environment:
```bash
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) Install Jupyter notebook dependencies:
```bash
pip install jupyter ipykernel
```

### Method 2: Using pyproject.toml (Recommended)

1. Create a virtual environment:
```bash
# Create a new virtual environment
python -m venv venv
```

2. Activate the virtual environment:
```bash
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

3. Install the project in editable mode:
```bash
pip install -e .
```

4. (Optional) Install development dependencies:
```bash
pip install -e ".[dev]"
```

5. (Optional) Install Jupyter notebook dependencies:
```bash
pip install -e ".[notebook]"
```

## Development

The project uses Hatch for development tasks. After installing the project, you can use the following commands:

```bash
# Run tests
hatch run test

# Run tests with coverage
hatch run test-cov

# Run linting
hatch run lint

# Run type checking
hatch run type
```

### Pre-commit Hooks

The project uses pre-commit hooks to ensure code quality. To set up pre-commit hooks:

1. Install pre-commit:
```bash
pip install pre-commit
```

2. Install the git hook scripts:
```bash
pre-commit install
```

The pre-commit hooks will run automatically on every commit and check for:
- Code formatting (black)
- Import sorting (isort)
- Linting (flake8)
- Type checking (mypy)
- Security issues (bandit)
- Common code issues (pre-commit hooks)

To run all hooks manually:
```bash
pre-commit run --all-files
```

## Testing

The project includes a comprehensive test suite. You can run tests in several ways:

1. Using pytest directly:
```bash
python -m pytest tests/
```

2. Using Hatch (recommended):
```bash
hatch run test
```

3. Run tests with coverage:
```bash
hatch run test-cov
```

The test suite includes:
- Pipeline tests
- Query processor tests
- Query decomposer tests
- Retriever tests
- Reranker tests
- Generator tests

## Usage

### Data Ingestion

```python
from src import DataIngestionPipeline

# Initialize the pipeline
ingestion_pipeline = DataIngestionPipeline("path/to/documents")

# Process documents
results = ingestion_pipeline.process_folder()
```

### Query Processing

```python
from src import QueryPipeline

# Initialize the pipeline
query_pipeline = QueryPipeline("path/to/documents")

# Process a query
result = query_pipeline.process_query(
    query="Your question here",
    n_results=5
)
```

### Interactive Demo

Run the Jupyter notebook `rag_pipeline_demo.ipynb` for an interactive demonstration.

## Features

- Automatic document processing and chunking
- Intelligent query decomposition
- Context-aware document retrieval
- Result reranking
- Natural language answer generation

## License

MIT License 