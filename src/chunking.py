import pandas as pd
import re
from typing import List, Dict
from nltk.tokenize import sent_tokenize
import nltk
from config import SENTENCES_PER_CHUNK, OVERLAP_SENTENCES

class TextChunker:
    def __init__(self):
        """
        Initialize the TextChunker with configuration from config.py.
        """
        # Download NLTK punkt tokenizer if not already downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using NLTK's sent_tokenize.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of sentences
        """
        # Use NLTK's sentence tokenizer
        sentences = sent_tokenize(text)
        # Filter out empty sentences and strip whitespace
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _create_chunks(self, sentences: List[str]) -> List[str]:
        """
        Create overlapping chunks from a list of sentences.
        
        Args:
            sentences (List[str]): List of sentences
            
        Returns:
            List[str]: List of text chunks
        """
        chunks = []
        step = SENTENCES_PER_CHUNK - OVERLAP_SENTENCES
        
        for i in range(0, len(sentences), step):
            chunk = sentences[i:i + SENTENCES_PER_CHUNK]
            if len(chunk) >= SENTENCES_PER_CHUNK - OVERLAP_SENTENCES:  # Only add chunks with sufficient content
                chunks.append(' '.join(chunk))
        
        return chunks
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the input DataFrame and create chunks for each document.
        
        Args:
            df (pd.DataFrame): Input DataFrame with columns ['company', 'service', 'file_name', 'text']
            
        Returns:
            pd.DataFrame: Output DataFrame with columns ['company', 'service', 'file_name', 'chunk']
        """
        result_data = []
        
        for _, row in df.iterrows():
            company = row['company']
            service = row['service']
            file_name = row['file_name']
            text = row['text']
            
            # Split text into sentences
            sentences = self._split_into_sentences(text)
            
            # Create chunks
            chunks = self._create_chunks(sentences)
            
            # Add each chunk to the result
            for chunk in chunks:
                result_data.append({
                    'company': company,
                    'service': service,
                    'file_name': file_name,
                    'chunk': chunk
                })
        
        return pd.DataFrame(result_data) 