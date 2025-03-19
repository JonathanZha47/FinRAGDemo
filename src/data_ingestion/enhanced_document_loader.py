from typing import List, Optional, Union
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter
import tabula
from pathlib import Path
import os

class EnhancedDocumentLoader:
    """Enhanced document loader with table extraction and text splitting."""
    
    def __init__(self):
        self.sentence_splitter = SentenceSplitter(
            chunk_size=1024,
            chunk_overlap=20,
            separator=" ",
            paragraph_separator="\n\n"
        )
    
    def load_documents(self, input_path: Union[str, Path]) -> List[Document]:
        """Load and process a single document or all documents in a directory."""
        documents = []
        input_path = Path(input_path)
        
        try:
            if input_path.is_file():
                docs = self._process_file(input_path)
                if docs:
                    documents.extend(docs)
            else:
                for file_path in input_path.glob("**/*"):
                    if file_path.suffix.lower() in ['.pdf', '.txt', '.docx']:
                        print(f"Processing {file_path.name}...")
                        docs = self._process_file(file_path)
                        if docs:
                            documents.extend(docs)
            
            if not documents:
                print("No documents were successfully processed")
                return []
            
            return self._split_documents(documents)
        except Exception as e:
            print(f"Error in load_documents: {str(e)}")
            return []
    
    def _process_file(self, file_path: Path) -> List[Document]:
        """Process a single file based on its type."""
        try:
            if not file_path.exists():
                print(f"File not found: {file_path}")
                return []
                
            if file_path.suffix.lower() == '.pdf':
                return self._process_pdf(file_path)
            else:
                try:
                    docs = SimpleDirectoryReader(input_files=[str(file_path)]).load_data()
                    return docs if docs else []
                except Exception as e:
                    print(f"Error reading file {file_path}: {str(e)}")
                    return []
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            return []
    
    def _process_pdf(self, pdf_path: Path) -> List[Document]:
        """Process PDF file with table extraction."""
        documents = []
        
        try:
            # 1. Extract text content
            try:
                base_docs = SimpleDirectoryReader(input_files=[str(pdf_path)]).load_data()
                if base_docs:
                    documents.extend(base_docs)
            except Exception as e:
                print(f"Text extraction error for {pdf_path}: {str(e)}")
            
            # 2. Extract tables
            try:
                tables = tabula.read_pdf(str(pdf_path), pages='all', multiple_tables=True)
                if tables:
                    for i, table in enumerate(tables):
                        if not table.empty:
                            table_text = table.to_markdown(index=False)
                            table_doc = Document(
                                text=f"Table {i+1}:\n{table_text}",
                                metadata={"source": str(pdf_path), "type": "table", "table_index": i}
                            )
                            documents.append(table_doc)
            except Exception as e:
                print(f"Table extraction error for {pdf_path}: {str(e)}")
            
            # 3. OCR Placeholder for future implementation
            # TODO: Implement OCR processing in future iterations
            
            if not documents:
                print(f"Warning: No content extracted from {pdf_path}")
            
            return documents
            
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {str(e)}")
            return []
    
    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks using the sentence splitter."""
        try:
            if not documents:
                return []
                
            split_docs = []
            for doc in documents:
                if not doc.text:
                    continue
                    
                chunks = self.sentence_splitter.split_text(doc.text)
                for i, chunk in enumerate(chunks):
                    if chunk.strip():  # Only add non-empty chunks
                        metadata = doc.metadata.copy() if doc.metadata else {}
                        metadata['chunk_index'] = i
                        split_docs.append(Document(text=chunk, metadata=metadata))
            
            return split_docs
        except Exception as e:
            print(f"Error splitting documents: {str(e)}")
            return documents 