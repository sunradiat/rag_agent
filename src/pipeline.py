from typing import Dict, List, Optional
from query_processor import QueryProcessor
from query_decomposer import QueryDecomposer
from retriever import Retriever
from reranker import Reranker
from generation import Generator
from database_operation import DatabaseManager
from chunking import TextChunker
from embedding import EmbeddingGenerator
import os
from .query_processor_graph import QueryProcessorGraph

class DataIngestionPipeline:
    def __init__(self, input_folder: str):
        """
        Initialize the data ingestion pipeline.
        
        Args:
            input_folder (str): Path to the input folder containing documents
        """
        self.input_folder = input_folder
        self.chunker = TextChunker()
        self.embedder = EmbeddingGenerator()
        self.db_manager = DatabaseManager(input_folder)
    
    def process_document(self, file_path: str) -> Dict:
        """
        Process a single document through the ingestion pipeline.
        
        Args:
            file_path (str): Path to the document file
            
        Returns:
            Dict: Processing results
        """
        # Read document
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Chunk text
        chunks = self.chunker.create_chunks(text)
        
        # Generate embeddings
        embeddings = self.embedder.generate_embeddings([chunk['text'] for chunk in chunks])
        
        # Prepare metadata
        metadata = []
        for i, chunk in enumerate(chunks):
            metadata.append({
                'file_name': os.path.basename(file_path),
                'chunk_id': i,
                'text': chunk['text']
            })
        
        # Store in database
        self.db_manager.store_embeddings(embeddings, metadata)
        
        return {
            'file_name': os.path.basename(file_path),
            'total_chunks': len(chunks),
            'status': 'success'
        }
    
    def process_folder(self) -> List[Dict]:
        """
        Process all documents in the input folder.
        
        Returns:
            List[Dict]: List of processing results for each document
        """
        results = []
        for filename in os.listdir(self.input_folder):
            if filename.endswith('.txt'):
                file_path = os.path.join(self.input_folder, filename)
                result = self.process_document(file_path)
                results.append(result)
        return results

class QueryPipeline:
    def __init__(self, input_folder: str):
        """
        Initialize the query processing pipeline.
        
        Args:
            input_folder (str): Path to the input folder containing the database
        """
        self.processor = QueryProcessorGraph(input_folder)
    
    def process_query(self, query: str, n_results: int = 5) -> Dict:
        """
        Process a query through the pipeline.
        
        Args:
            query (str): User query
            n_results (int): Number of results to return
            
        Returns:
            Dict: Processing results including answer and sources
        """
        return self.processor.process_query(query) 