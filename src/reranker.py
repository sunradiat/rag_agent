from typing import List, Dict
import google.cloud.aiplatform as aiplatform
from config import PROJECT_ID, LOCATION
from embedding import EmbeddingGenerator
from vertexai.preview.generative_models import GenerativeModel

class Reranker:
    def __init__(self):
        """
        Initialize the Reranker with Vertex AI.
        """
        # Initialize Vertex AI
        aiplatform.init(project=PROJECT_ID, location=LOCATION)
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator()
        
        # Initialize Gemini model for reranking
        self.model = GenerativeModel("gemini-pro")
    
    def _create_rerank_prompt(self, query: str, document: str) -> str:
        """
        Create a prompt for reranking a document.
        
        Args:
            query (str): Original query
            document (str): Document content
            
        Returns:
            str: Prompt for reranking
        """
        return f"""You are a document relevance evaluator. Your task is to score how well a document answers a given query.
Score should be between 0 and 1, where 1 means the document perfectly answers the query and 0 means the document is completely irrelevant.

Query: {query}

Document: {document}

Please provide a single number between 0 and 1 representing the relevance score. Do not include any explanation or additional text.
"""
    
    def rerank(self, query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Rerank the retrieved documents using cross-encoder approach with Gemini.
        
        Args:
            query (str): Original query
            results (List[Dict]): Retrieved documents from the retriever
            top_k (int): Number of top results to return after reranking
            
        Returns:
            List[Dict]: Reranked documents
        """
        if not results:
            return []
        
        try:
            # Score each document
            scored_results = []
            for result in results:
                # Create prompt for this document
                prompt = self._create_rerank_prompt(query, result['document'])
                
                # Get relevance score
                response = self.model.generate_content(prompt)
                try:
                    score = float(response.text.strip())
                    # Ensure score is between 0 and 1
                    score = max(0.0, min(1.0, score))
                except (ValueError, AttributeError):
                    # If score parsing fails, use original score
                    score = result.get('score', 0.0)
                
                # Create new result with rerank score
                new_result = result.copy()
                new_result['rerank_score'] = score
                scored_results.append(new_result)
            
            # Sort by rerank score
            scored_results.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            # Return top k results
            return scored_results[:top_k]
            
        except Exception as e:
            print(f"Error in reranking: {str(e)}")
            return results  # Return original results if reranking fails 