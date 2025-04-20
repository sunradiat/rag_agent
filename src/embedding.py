import pandas as pd
from typing import List, Dict
import numpy as np
from google.cloud import aiplatform
from vertexai.preview.language_models import TextEmbeddingModel
from config import PROJECT_ID, LOCATION, BATCH_SIZE, EMBEDDING_MODEL

class EmbeddingGenerator:
    def __init__(self):
        """
        Initialize the EmbeddingGenerator with configuration from config.py.
        """
        # Initialize Vertex AI
        aiplatform.init(project=PROJECT_ID, location=LOCATION)
        
        # Initialize the embedding model
        self.model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
        self.batch_size = BATCH_SIZE
    
    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text using Vertex AI's text embedding model.
        
        Args:
            text (str): Input text
            
        Returns:
            List[float]: Embedding vector
        """
        try:
            embeddings = self.model.get_embeddings([text])
            return embeddings[0].values
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            return None
    
    def _process_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Process a batch of texts to get their embeddings.
        
        Args:
            texts (List[str]): List of texts to process
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        try:
            # Get embeddings for the entire batch at once
            embeddings = self.model.get_embeddings(texts)
            return [embedding.values for embedding in embeddings]
        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            # If batch processing fails, fall back to individual processing
            embeddings = []
            for text in texts:
                embedding = self._get_embedding(text)
                if embedding is not None:
                    embeddings.append(embedding)
                else:
                    # Add a zero vector if embedding generation failed
                    embeddings.append([0.0] * 768)  # textembedding-gecko produces 768-dimensional vectors
            return embeddings
    
    def process_embedding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the input DataFrame and add embeddings for each chunk.
        
        Args:
            df (pd.DataFrame): Input DataFrame with columns including 'chunk'
            
        Returns:
            pd.DataFrame: Output DataFrame with additional 'embedding' column
        """
        # Create a copy of the input DataFrame
        result_df = df.copy()
        
        # Process chunks in batches
        chunks = df['chunk'].tolist()
        embeddings = []
        
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            batch_embeddings = self._process_batch(batch)
            embeddings.extend(batch_embeddings)
            
            # Print progress
            print(f"Processed {min(i + self.batch_size, len(chunks))}/{len(chunks)} chunks")
        
        # Add embeddings to the DataFrame
        result_df['embedding'] = embeddings
        
        return result_df
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        
        Args:
            query (str): Search query text
            
        Returns:
            List[float]: Query embedding vector
        """
        try:
            embedding = self._get_embedding(query)
            if embedding is None:
                print("Failed to generate query embedding")
                return [0.0] * 768  # Return zero vector if embedding generation failed
            return embedding
        except Exception as e:
            print(f"Error embedding query: {str(e)}")
            return [0.0] * 768  # Return zero vector if any error occurs 