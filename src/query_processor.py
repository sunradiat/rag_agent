from typing import List, Dict, Optional
from vertexai.preview.generative_models import GenerativeModel
import google.cloud.aiplatform as aiplatform
from config import PROJECT_ID, LOCATION
from query_decomposer import QueryDecomposer

class QueryProcessor:
    def __init__(self):
        """
        Initialize the QueryProcessor with Gemini model for query rewriting
        and QueryDecomposer for complex query handling.
        """
        # Initialize Vertex AI
        aiplatform.init(project=PROJECT_ID, location=LOCATION)
        
        # Initialize Gemini model
        self.model = GenerativeModel("gemini-pro")
        
        # Initialize QueryDecomposer
        self.decomposer = QueryDecomposer()
    
    def rewrite(self, query: str, company: Optional[str] = None, service: Optional[str] = None) -> Dict:
        """
        Rewrite the user query to improve retrieval effectiveness.
        
        Args:
            query (str): Original user query
            company (Optional[str]): Company context
            service (Optional[str]): Service context
            
        Returns:
            Dict: Dictionary containing rewritten queries and explanations
        """
        # First analyze query complexity
        is_complex, complexity_explanation = self.decomposer.analyze_complexity(query)
        
        if is_complex:
            # Decompose the complex query
            decomposed = self.decomposer.decompose(query)
            return {
                "is_complex": True,
                "complexity_explanation": complexity_explanation,
                "decomposition": decomposed,
                "main_query": query,
                "alternative_queries": [],
                "explanation": "Query was decomposed into sub-questions due to complexity."
            }
        else:
            # Regular query rewriting
            prompt = f"""You are a query rewriting expert. Your task is to improve the given query for better document retrieval.
The rewritten queries should:
1. Maintain the original intent
2. Include relevant synonyms and related terms
3. Clarify ambiguous terms
4. Consider the business context if provided

Original Query: {query}
{f"Company Context: {company}" if company else ""}
{f"Service Context: {service}" if service else ""}

Please provide:
1. A main rewritten query (most comprehensive version)
2. 2-3 alternative queries (different phrasings or focuses)
3. A brief explanation of the changes made

Format your response as a JSON object with the following structure:
{{
    "main_query": "main rewritten query",
    "alternative_queries": ["alternative query 1", "alternative query 2"],
    "explanation": "brief explanation of the changes"
}}
"""
            
            try:
                response = self.model.generate_content(prompt)
                import json
                result = json.loads(response.text)
                return {
                    "is_complex": False,
                    "complexity_explanation": complexity_explanation,
                    "decomposition": None,
                    **result
                }
            except Exception as e:
                print(f"Error in query rewriting: {str(e)}")
                return {
                    "is_complex": False,
                    "complexity_explanation": complexity_explanation,
                    "decomposition": None,
                    "main_query": query,
                    "alternative_queries": [query],
                    "explanation": f"Error occurred: {str(e)}"
                }
    
    def process_and_retrieve(self, query: str, retriever, company: Optional[str] = None, 
                           service: Optional[str] = None, n_results: int = 5) -> Dict:
        """
        Process the query (analyze, decompose or rewrite) and retrieve documents.
        
        Args:
            query (str): Original user query
            retriever: Retriever instance
            company (Optional[str]): Company context
            service (Optional[str]): Service context
            n_results (int): Number of results to return per query
            
        Returns:
            Dict: Dictionary containing processed query info and combined results
        """
        # Analyze and process the query
        processed = self.rewrite(query, company, service)
        
        if processed["is_complex"]:
            # For complex queries, retrieve for each sub-question
            all_results = []
            seen_docs = set()
            
            for sub_query in processed["decomposition"]["sub_questions"]:
                sub_results = retriever.retrieve(
                    query=sub_query,
                    company=company,
                    service=service,
                    n_results=n_results
                )
                
                # Add unique results
                for result in sub_results:
                    doc_id = result['metadata']['file_name']
                    if doc_id not in seen_docs:
                        seen_docs.add(doc_id)
                        all_results.append(result)
            
            return {
                "processed_query": processed,
                "results": all_results
            }
        else:
            # For simple queries, use the original rewrite_and_retrieve logic
            all_results = []
            seen_docs = set()
            
            # Retrieve using main query
            main_results = retriever.retrieve(
                query=processed["main_query"],
                company=company,
                service=service,
                n_results=n_results
            )
            
            # Add unique results
            for result in main_results:
                doc_id = result['metadata']['file_name']
                if doc_id not in seen_docs:
                    seen_docs.add(doc_id)
                    all_results.append(result)
            
            # Retrieve using alternative queries
            for alt_query in processed["alternative_queries"]:
                alt_results = retriever.retrieve(
                    query=alt_query,
                    company=company,
                    service=service,
                    n_results=n_results
                )
                
                # Add unique results
                for result in alt_results:
                    doc_id = result['metadata']['file_name']
                    if doc_id not in seen_docs:
                        seen_docs.add(doc_id)
                        all_results.append(result)
            
            return {
                "processed_query": processed,
                "results": all_results
            }
            
    # For backward compatibility
    rewrite_and_retrieve = process_and_retrieve 