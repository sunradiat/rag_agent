# Google Cloud configuration
PROJECT_ID = "your-project-id"  # Replace with your Google Cloud project ID
LOCATION = "us-central1"        # Vertex AI location/region

# Embedding configuration
BATCH_SIZE = 10                 # Number of texts to process in each batch
EMBEDDING_MODEL = "textembedding-gecko@001"  # Vertex AI embedding model

# Chunking configuration
SENTENCES_PER_CHUNK = 10        # Number of sentences in each chunk
OVERLAP_SENTENCES = 3           # Number of overlapping sentences between chunks 