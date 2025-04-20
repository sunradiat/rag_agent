import os
import chromadb
from chromadb.config import Settings
import pandas as pd
from typing import Optional, List, Dict
import json

class DatabaseManager:
    def __init__(self, input_folder: str):
        """
        Initialize the DatabaseManager with the input folder path.
        
        Args:
            input_folder (str): Path to the input folder where ChromaDB will be stored
        """
        self.input_folder = input_folder
        self.chroma_folder = os.path.join(input_folder, "chromadb")
        
        # Create chromadb folder if it doesn't exist
        os.makedirs(self.chroma_folder, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.chroma_folder,
            settings=Settings(
                anonymized_telemetry=False
            )
        )
        
        # Create or get the collection
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
    
    def _prepare_metadata(self, row: pd.Series) -> Dict:
        """
        Prepare metadata for a document chunk.
        
        Args:
            row (pd.Series): Row from the DataFrame
            
        Returns:
            Dict: Metadata dictionary
        """
        return {
            "company": row['company'],
            "service": row['service'],
            "file_name": row['file_name']
        }
    
    def store_embeddings(self, df: pd.DataFrame) -> None:
        """
        Store document embeddings and metadata in ChromaDB.
        
        Args:
            df (pd.DataFrame): DataFrame containing embeddings and metadata
        """
        # Prepare data for ChromaDB
        ids = [f"doc_{i}" for i in range(len(df))]
        embeddings = df['embedding'].tolist()
        documents = df['chunk'].tolist()
        metadatas = [self._prepare_metadata(row) for _, row in df.iterrows()]
        
        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        # Save metadata mapping for later use
        self._save_metadata_mapping(df)
    
    def _save_metadata_mapping(self, df: pd.DataFrame) -> None:
        """
        Save metadata mapping to a JSON file for later reference.
        
        Args:
            df (pd.DataFrame): DataFrame containing metadata
        """
        mapping = {
            "companies": sorted(df['company'].unique().tolist()),
            "services": sorted(df['service'].unique().tolist()),
            "file_names": sorted(df['file_name'].unique().tolist())
        }
        
        mapping_path = os.path.join(self.chroma_folder, "metadata_mapping.json")
        with open(mapping_path, 'w') as f:
            json.dump(mapping, f, indent=2)
    
    def get_available_filters(self) -> Dict:
        """
        Get available companies and services for filtering.
        
        Returns:
            Dict: Dictionary containing available companies and services
        """
        mapping_path = os.path.join(self.chroma_folder, "metadata_mapping.json")
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r') as f:
                return json.load(f)
        return {"companies": [], "services": [], "file_names": []} 