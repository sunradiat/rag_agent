from typing import Dict, Tuple
from vertexai.preview.generative_models import GenerativeModel
import google.cloud.aiplatform as aiplatform
from config import PROJECT_ID, LOCATION

class QueryDecomposer:
    def __init__(self):
        """
        Initialize the QueryDecomposer with Gemini model for query analysis and decomposition.
        """
        # Initialize Vertex AI
        aiplatform.init(project=PROJECT_ID, location=LOCATION)
        
        # Initialize Gemini model
        self.model = GenerativeModel("gemini-pro")
    
    def analyze_complexity(self, query: str) -> Tuple[bool, str]:
        """
        Analyze if the query is complex and needs decomposition.
        
        Args:
            query (str): User query
            
        Returns:
            Tuple[bool, str]: (is_complex, explanation)
        """
        prompt = f"""Analyze if the following query is complex and needs decomposition.
A complex query typically:
1. Contains multiple sub-questions or aspects
2. Requires information from different domains
3. Needs step-by-step reasoning
4. Has dependencies between different parts

Query: {query}

Please provide a JSON response with:
1. is_complex: boolean indicating if the query is complex
2. explanation: brief explanation of why it is or isn't complex

Example response format:
{{
    "is_complex": true,
    "explanation": "The query contains multiple aspects about cloud computing that need to be addressed separately."
}}
"""
        
        try:
            response = self.model.generate_content(prompt)
            import json
            result = json.loads(response.text)
            return result["is_complex"], result["explanation"]
        except Exception as e:
            print(f"Error in complexity analysis: {str(e)}")
            return False, "Error in analysis"
    
    def decompose(self, query: str) -> Dict:
        """
        Decompose a complex query into sub-questions.
        
        Args:
            query (str): Complex user query
            
        Returns:
            Dict: Dictionary containing decomposed queries and structure
        """
        prompt = f"""Decompose the following complex query into logical sub-questions.
The decomposition should:
1. Break down the main question into clear, focused sub-questions
2. Maintain the logical flow and dependencies
3. Ensure each sub-question is answerable independently
4. Consider the relationships between sub-questions

Query: {query}

Please provide a JSON response with:
1. main_question: the original query
2. sub_questions: list of decomposed sub-questions
3. structure: description of how sub-questions relate to each other
4. dependencies: list showing which sub-questions depend on others

Example response format:
{{
    "main_question": "What are the benefits and challenges of implementing cloud computing?",
    "sub_questions": [
        "What are the main benefits of cloud computing?",
        "What are the key challenges in cloud implementation?",
        "How do these benefits and challenges relate to each other?"
    ],
    "structure": "The first two questions explore separate aspects, while the third connects them.",
    "dependencies": [
        {{"question": 2, "depends_on": [0, 1]}}
    ]
}}
"""
        
        try:
            response = self.model.generate_content(prompt)
            import json
            return json.loads(response.text)
        except Exception as e:
            print(f"Error in query decomposition: {str(e)}")
            return {
                "main_question": query,
                "sub_questions": [query],
                "structure": "Error in decomposition",
                "dependencies": []
            } 