import os
from pathlib import Path
from typing import List, Optional, Union
from llama_index.core import SimpleDirectoryReader
from src.config.config_loader import ConfigLoader
from src.data_ingestion.enhanced_document_loader import EnhancedDocumentLoader

class DocumentLoader:
    """Document loader for ingesting various file formats with table extraction."""
    
    def __init__(self, config_loader: ConfigLoader):
        self.config = config_loader.get_data_config()
        self.input_dir = self.config.get('input_dir', 'data')
        self.supported_formats = self.config.get('supported_formats', ['.txt', '.pdf', '.docx'])
        
        # Initialize enhanced document loader
        self.enhanced_loader = EnhancedDocumentLoader()
    
    def load_single_document(self, file_path: Union[str, Path]) -> List:
        """Load and process a single document."""
        try:
            file_path = Path(file_path)
            
            # Validate file format
            if not self.validate_file_format(file_path):
                print(f"Unsupported file format: {file_path.suffix}")
                return []
            
            # Process the file using enhanced loader
            documents = self.enhanced_loader.load_documents(file_path)
            
            if not documents:
                print("No content was extracted from the document")
                return []
            
            print(f"Successfully processed document: {file_path.name}")
            return documents
            
        except Exception as e:
            print(f"Error processing document {file_path}: {str(e)}")
            return []
    
    def load_documents(self, directory: Optional[str] = None) -> List:
        """Load documents from specified directory."""
        try:
            target_dir = directory or self.input_dir
            
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
                print(f"Created directory {target_dir}, but no files found.")
                return []
            
            if not os.listdir(target_dir):
                print("No files found in the data directory")
                return []
            
            documents = self.enhanced_loader.load_documents(target_dir)
            if not documents:
                print("No documents were loaded")
                return []
            
            print(f"Successfully loaded {len(documents)} documents")
            return documents
            
        except Exception as e:
            print(f"Error loading documents: {str(e)}")
            return []
    
    def validate_file_format(self, file_path: Union[str, Path]) -> bool:
        """Validate if file format is supported."""
        return Path(file_path).suffix.lower() in self.supported_formats 