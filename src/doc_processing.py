import os
import pandas as pd
import fitz  # PyMuPDF
from docx import Document
import pytesseract
from pdf2image import convert_from_path

class DocumentProcessor:
    def __init__(self, folder_path: str):
        """
        Initialize the DocumentProcessor with the folder path.
        
        Args:
            folder_path (str): Absolute path to the folder containing documents
        """
        self.folder_path = os.path.abspath(folder_path)
        self.df = pd.DataFrame(columns=['company', 'service', 'file_name', 'text'])
        
    def _is_scanned_pdf(self, pdf_path: str) -> bool:
        """
        Check if a PDF is scanned (contains images) by attempting to extract text.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            bool: True if the PDF appears to be scanned
        """
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                if page.get_text().strip():
                    return False
            return True
        except:
            return True
    
    def _process_pdf(self, pdf_path: str) -> str:
        """
        Process a regular PDF file and extract text using PyMuPDF.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text from the PDF
        """
        text = ""
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text += page.get_text() + "\n"
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {str(e)}")
        return text.strip()
    
    def _process_scanned_pdf(self, pdf_path: str) -> str:
        """
        Process a scanned PDF using OCR.
        
        Args:
            pdf_path (str): Path to the scanned PDF file
            
        Returns:
            str: Extracted text from the scanned PDF
        """
        text = ""
        try:
            # Convert PDF pages to images
            images = convert_from_path(pdf_path)
            
            # Process each page with OCR
            for image in images:
                text += pytesseract.image_to_string(image) + "\n"
        except Exception as e:
            print(f"Error processing scanned PDF {pdf_path}: {str(e)}")
        return text.strip()
    
    def _process_docx(self, docx_path: str) -> str:
        """
        Process a DOCX file and extract text.
        
        Args:
            docx_path (str): Path to the DOCX file
            
        Returns:
            str: Extracted text from the DOCX
        """
        text = ""
        try:
            doc = Document(docx_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            print(f"Error processing DOCX {docx_path}: {str(e)}")
        return text.strip()
    
    def _extract_metadata_from_path(self, file_path: str) -> tuple:
        """
        Extract company and service information from the file path.
        Handles both contract folder with company subfolders and flat folder structures.
        
        Args:
            file_path (str): Full path to the file
            
        Returns:
            tuple: (company, service) extracted from the path
        """
        # Get the relative path from the input folder
        rel_path = os.path.relpath(file_path, self.folder_path)
        path_parts = rel_path.split(os.sep)
        
        # Check if we're in a contract folder structure (has company subfolders)
        if len(path_parts) > 1 and os.path.isdir(os.path.join(self.folder_path, path_parts[0])):
            # Contract folder structure: contract_folder/company/service/file
            company = path_parts[0]  # First level is company
            service = path_parts[1] if len(path_parts) > 1 else "Unknown"
        else:
            # Flat folder structure
            company = "Unknown"
            service = "Unknown"
            
            # Try to extract company and service from filename
            filename = os.path.splitext(os.path.basename(file_path))[0]
            parts = filename.split('_')
            if len(parts) >= 2:
                company = parts[0]
                service = parts[1]
        
        return company, service
    
    def process_folder(self) -> pd.DataFrame:
        """
        Process all documents in the folder and return a DataFrame with results.
        Handles both contract folder with company subfolders and flat folder structures.
        
        Returns:
            pd.DataFrame: DataFrame containing processed document information
        """
        for root, _, files in os.walk(self.folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                try:
                    if file_ext == '.pdf':
                        # Check if it's a scanned PDF
                        if self._is_scanned_pdf(file_path):
                            text = self._process_scanned_pdf(file_path)
                        else:
                            text = self._process_pdf(file_path)
                    elif file_ext == '.docx':
                        text = self._process_docx(file_path)
                    else:
                        print(f"Unsupported file type: {file}")
                        continue
                    
                    # Extract company and service information
                    company, service = self._extract_metadata_from_path(file_path)
                    
                    # Add to DataFrame
                    new_row = {
                        'company': company,
                        'service': service,
                        'file_name': file,
                        'text': text
                    }
                    self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
                    
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")
                    
        return self.df

processor = DocumentProcessor("path/to/your/documents")
df = processor.process_folder()
print(df)