from typing import List, Dict, Tuple
from vertexai.preview.generative_models import GenerativeModel
import google.cloud.aiplatform as aiplatform
from config import PROJECT_ID, LOCATION
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

class Generator:
    def __init__(self):
        """
        Initialize the Generator with Gemini model.
        """
        # Initialize Vertex AI
        aiplatform.init(project=PROJECT_ID, location=LOCATION)
        
        # Initialize Gemini model
        self.model = GenerativeModel("gemini-pro")
        
        # Download NLTK data if not already downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def _format_context(self, results: List[Dict]) -> str:
        """
        Format retrieved documents into context for the model.
        
        Args:
            results (List[Dict]): List of retrieved documents
            
        Returns:
            str: Formatted context string
        """
        context = "Here are the relevant documents:\n\n"
        
        for i, result in enumerate(results, 1):
            context += f"Document {i}:\n"
            context += f"Source: {result['metadata']['file_name']}\n"
            context += f"Content: {result['document']}\n\n"
        
        return context
    
    def _verify_answer(self, answer: str, results: List[Dict]) -> Tuple[bool, List[str]]:
        """
        Verify if the answer is based on the provided documents.
        
        Args:
            answer (str): Generated answer
            results (List[Dict]): Retrieved documents
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, missing_points)
        """
        # Tokenize answer and documents
        answer_tokens = set(word_tokenize(answer.lower()))
        doc_tokens = set()
        for result in results:
            doc_tokens.update(word_tokenize(result['document'].lower()))
        
        # Find unique tokens in answer that are not in documents
        # Exclude common words and punctuation
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        missing_tokens = answer_tokens - doc_tokens - common_words
        
        # If there are significant missing tokens, the answer might contain fabricated information
        if len(missing_tokens) > 3:  # Allow for some flexibility
            return False, list(missing_tokens)
        
        return True, []
    
    def generate_answer(self, query: str, results: List[Dict]) -> str:
        """
        Generate an answer based on the query and retrieved documents.
        
        Args:
            query (str): Original query
            results (List[Dict]): Retrieved documents
            
        Returns:
            str: Generated answer
        """
        # Format the context from retrieved documents
        context = self._format_context(results)
        
        # Create the prompt
        prompt = f"""You are a helpful assistant that answers questions based on the provided documents.
Your answer should ONLY use information from the provided documents. Do not make up or add any information that is not in the documents.

{context}

Question: {query}

Please provide a clear and concise answer based ONLY on the information in the provided documents.
If the documents do not contain enough information to answer the question, say "I cannot answer this question based on the provided documents."
"""
        
        try:
            # Generate response
            response = self.model.generate_content(prompt)
            answer = response.text
            
            # Verify the answer
            is_valid, missing_points = self._verify_answer(answer, results)
            if not is_valid:
                return "I cannot provide a reliable answer based on the provided documents. The generated answer contains information not found in the source documents."
            
            return answer
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def generate_answer_with_sources(self, query: str, results: List[Dict]) -> Dict:
        """
        Generate an answer with source documents.
        
        Args:
            query (str): Original query
            results (List[Dict]): Retrieved documents
            
        Returns:
            Dict: Dictionary containing answer and sources
        """
        # Generate the answer
        answer = self.generate_answer(query, results)
        
        # Prepare sources (only file names)
        sources = [result['metadata']['file_name'] for result in results]
        
        return {
            'answer': answer,
            'sources': sources
        }
    
    def combine_answers(self, original_query: str, sub_answers: List[Dict]) -> Dict:
        """
        Combine answers from sub-questions into a final answer.
        
        Args:
            original_query (str): The original user query
            sub_answers (List[Dict]): List of answers for sub-questions
            
        Returns:
            Dict: Combined answer with sources
        """
        # Prepare the prompt
        prompt = f"""You are an expert at combining answers from multiple sub-questions into a coherent final answer.

Original Query: {original_query}

Sub-questions and their answers:
{chr(10).join([f"Q: {answer['question']}\nA: {answer['answer']}" for answer in sub_answers])}

Please combine these answers into a comprehensive final answer that directly addresses the original query.
Make sure to:
1. Maintain a logical flow
2. Avoid repetition
3. Ensure all key points are covered
4. Make the answer sound natural and coherent

Final Answer:"""

        # Generate the combined answer
        response = self.model.generate_content(prompt)
        
        # Collect all unique sources
        all_sources = set()
        for answer in sub_answers:
            all_sources.update(answer['sources'])
        
        return {
            "answer": response.text,
            "sources": list(all_sources)
        } 