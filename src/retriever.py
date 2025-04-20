import chromadb
from chromadb.config import Settings
from typing import Optional, List, Dict, Tuple
import os
from rank_bm25 import BM25Okapi
import numpy as np
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
from embedding import EmbeddingGenerator

class Retriever:
    def __init__(self, input_folder: str):
        """
        Initialize the Retriever with the input folder path.
        
        Args:
            input_folder (str): Path to the input folder containing ChromaDB
        """
        self.input_folder = input_folder
        self.chroma_folder = os.path.join(input_folder, "chromadb")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.chroma_folder,
            settings=Settings(
                anonymized_telemetry=False
            )
        )
        
        # Get the collection
        self.collection = self.client.get_collection("documents")
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator()
        
        # Download NLTK data if not already downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # Initialize BM25
        self._initialize_bm25()
    
    def _initialize_bm25(self) -> None:
        """
        Initialize BM25 with all documents in the collection.
        """
        # Get all documents
        results = self.collection.get(
            include=["documents", "metadatas"]
        )
        
        # Prepare documents for BM25
        self.documents = results['documents']
        self.metadatas = results['metadatas']
        
        # Tokenize documents for BM25 using NLTK
        tokenized_docs = [word_tokenize(doc.lower()) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def _build_where_clause(self, company: Optional[str] = None, service: Optional[str] = None) -> Dict:
        """
        Build the where clause for filtering.
        
        Args:
            company (Optional[str]): Company to filter by
            service (Optional[str]): Service to filter by
            
        Returns:
            Dict: Where clause for filtering
        """
        where_clause = {}
        
        if company:
            where_clause["company"] = company
        if service:
            where_clause["service"] = service
            
        return where_clause if where_clause else None
    
    def _get_filtered_indices(self, where_clause: Optional[Dict] = None) -> List[int]:
        """
        Get indices of documents that match the filter criteria.
        
        Args:
            where_clause (Optional[Dict]): Filter criteria
            
        Returns:
            List[int]: List of document indices
        """
        if not where_clause:
            return list(range(len(self.documents)))
        
        filtered_indices = []
        for i, metadata in enumerate(self.metadatas):
            match = True
            for key, value in where_clause.items():
                if metadata.get(key) != value:
                    match = False
                    break
            if match:
                filtered_indices.append(i)
        return filtered_indices
    
    def dense_retrieve(self, query: str, company: Optional[str] = None, service: Optional[str] = None, 
                      n_results: int = 5) -> List[Dict]:
        """
        Retrieve documents using dense retrieval (embedding similarity).
        
        Args:
            query (str): Search query
            company (Optional[str]): Company to filter by
            service (Optional[str]): Service to filter by
            n_results (int): Number of results to return
            
        Returns:
            List[Dict]: List of retrieved documents with metadata
        """
        where_clause = self._build_where_clause(company, service)
        
        # Get query embedding using embed_query method
        query_embedding = self.embedding_generator.embed_query(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )
        
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'score': 1 - results['distances'][0][i]  # Convert distance to similarity score
            })
        
        return formatted_results
    
    def sparse_retrieve(self, query: str, company: Optional[str] = None, service: Optional[str] = None,
                       n_results: int = 5) -> List[Dict]:
        """
        Retrieve documents using sparse retrieval (BM25).
        
        Args:
            query (str): Search query
            company (Optional[str]): Company to filter by
            service (Optional[str]): Service to filter by
            n_results (int): Number of results to return
            
        Returns:
            List[Dict]: List of retrieved documents with metadata
        """
        where_clause = self._build_where_clause(company, service)
        filtered_indices = self._get_filtered_indices(where_clause)
        
        # Get BM25 scores using NLTK tokenization
        tokenized_query = word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokenized_query)
        
        # Filter scores and get top results
        filtered_scores = [(i, scores[i]) for i in filtered_indices]
        filtered_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in filtered_scores[:n_results]]
        
        # Format results
        formatted_results = []
        for idx in top_indices:
            formatted_results.append({
                'document': self.documents[idx],
                'metadata': self.metadatas[idx],
                'score': scores[idx]
            })
        
        return formatted_results
    
    def hybrid_retrieve(self, query: str, company: Optional[str] = None, service: Optional[str] = None,
                       n_results: int = 5) -> List[Dict]:
        """
        Retrieve documents using hybrid retrieval (equal weights for dense and sparse).
        
        Args:
            query (str): Search query
            company (Optional[str]): Company to filter by
            service (Optional[str]): Service to filter by
            n_results (int): Number of results to return
            
        Returns:
            List[Dict]: List of retrieved documents with metadata
        """
        # Get dense and sparse results
        dense_results = self.dense_retrieve(query, company, service, n_results * 2)
        sparse_results = self.sparse_retrieve(query, company, service, n_results * 2)
        
        # Create a dictionary to store combined scores
        combined_scores = defaultdict(float)
        doc_info = {}
        
        # Add dense scores
        for result in dense_results:
            doc = result['document']
            combined_scores[doc] += result['score'] * 0.5  # 50% weight
            doc_info[doc] = result
        
        # Add sparse scores
        for result in sparse_results:
            doc = result['document']
            combined_scores[doc] += result['score'] * 0.5  # 50% weight
            if doc not in doc_info:
                doc_info[doc] = result
        
        # Get top results
        top_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:n_results]
        
        # Format results
        formatted_results = []
        for doc, score in top_docs:
            result = doc_info[doc]
            result['score'] = score
            formatted_results.append(result)
        
        return formatted_results
    
    def retrieve(self, query: str, company: Optional[str] = None, service: Optional[str] = None,
                n_results: int = 5, method: str = "hybrid") -> List[Dict]:
        """
        Retrieve documents using specified method.
        
        Args:
            query (str): Search query
            company (Optional[str]): Company to filter by
            service (Optional[str]): Service to filter by
            n_results (int): Number of results to return
            method (str): Retrieval method ("dense", "sparse", or "hybrid")
            
        Returns:
            List[Dict]: List of retrieved documents with metadata
        """
        if method == "dense":
            return self.dense_retrieve(query, company, service, n_results)
        elif method == "sparse":
            return self.sparse_retrieve(query, company, service, n_results)
        elif method == "hybrid":
            return self.hybrid_retrieve(query, company, service, n_results)
        else:
            raise ValueError(f"Unknown retrieval method: {method}") 