"""
Data preprocessing module for loading and validating input data using LangChain.
"""

import os
import json
import csv
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("pandas not available. Some data loading features will be limited.")

try:
    from langchain_community.document_loaders import (
        TextLoader,
        JSONLoader,
        CSVLoader,
        PyPDFLoader,
        DirectoryLoader,
        DataFrameLoader
    )
    from langchain.schema import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("langchain not available. Using fallback data loading methods.")


class DataPreprocessor:
    """Handles data loading, validation, and format conversion."""
    
    def __init__(self, config):
        """
        Initialize the data preprocessor.
        
        Args:
            config: Configuration object containing preprocessing settings
        """
        self.config = config
        self.supported_formats = config.supported_formats
        self.encoding = config.encoding
        
    def load_data(self, input_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load data from various sources and formats using LangChain loaders.
        
        Args:
            input_path: Path to input data (file or directory)
            
        Returns:
            List of document dictionaries with 'content' and 'metadata' keys
        """
        input_path = input_path or self.config.input_data_path
        
        # Check if this is a Hugging Face dataset path
        if input_path.startswith("hf://"):
            return self._load_huggingface_dataset(input_path)
        
        path = Path(input_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")
        
        documents = []
        
        # Try to use LangChain loaders first
        if LANGCHAIN_AVAILABLE:
            try:
                if path.is_file():
                    documents = self._load_file_with_langchain(path)
                elif path.is_dir():
                    documents = self._load_directory_with_langchain(path)
            except Exception as e:
                logger.warning(f"LangChain loader failed: {e}. Falling back to basic loading.")
                if path.is_file():
                    documents = self._load_file(path)
                elif path.is_dir():
                    documents = self._load_directory(path)
        else:
            # Fallback to basic loading
            if path.is_file():
                documents = self._load_file(path)
            elif path.is_dir():
                documents = self._load_directory(path)
        
        logger.info(f"Loaded {len(documents)} documents from {input_path}")
        return documents
    
    def _load_directory(self, directory: Path) -> List[Dict[str, Any]]:
        """Load all supported files from a directory."""
        documents = []
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and self._is_supported_format(file_path):
                try:
                    docs = self._load_file(file_path)
                    documents.extend(docs)
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
        
        return documents
    
    def _load_file_with_langchain(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load a file using LangChain loaders."""
        extension = file_path.suffix.lower().lstrip('.')
        
        try:
            loader = None
            if extension == 'txt':
                loader = TextLoader(str(file_path), encoding=self.encoding)
            elif extension == 'json':
                loader = JSONLoader(str(file_path), jq_schema='.', text_content=False)
            elif extension == 'csv':
                loader = CSVLoader(str(file_path), encoding=self.encoding)
            elif extension == 'pdf':
                loader = PyPDFLoader(str(file_path))
            
            if loader:
                langchain_docs = loader.load()
                return self._convert_langchain_documents(langchain_docs)
        except Exception as e:
            logger.warning(f"LangChain loader failed for {file_path}: {e}")
            raise
        
        return []
    
    def _load_directory_with_langchain(self, directory: Path) -> List[Dict[str, Any]]:
        """Load directory using LangChain DirectoryLoader."""
        try:
            # Load text files
            documents = []
            for extension in self.supported_formats:
                if extension in ['txt', 'json', 'csv']:
                    glob_pattern = f"**/*.{extension}"
                    loader = DirectoryLoader(
                        str(directory),
                        glob=glob_pattern,
                        show_progress=False
                    )
                    try:
                        langchain_docs = loader.load()
                        documents.extend(self._convert_langchain_documents(langchain_docs))
                    except Exception as e:
                        logger.warning(f"Error loading {extension} files: {e}")
            
            return documents
        except Exception as e:
            logger.warning(f"DirectoryLoader failed: {e}")
            raise
    
    def _convert_langchain_documents(self, langchain_docs: List['Document']) -> List[Dict[str, Any]]:
        """Convert LangChain Documents to our document format."""
        documents = []
        for doc in langchain_docs:
            documents.append({
                'content': doc.page_content,
                'metadata': dict(doc.metadata) if doc.metadata else {}
            })
        return documents
    
    def _load_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load a single file based on its format."""
        extension = file_path.suffix.lower().lstrip('.')
        
        if extension == 'txt':
            return self._load_txt(file_path)
        elif extension == 'json':
            return self._load_json(file_path)
        elif extension == 'csv':
            return self._load_csv(file_path)
        elif extension == 'pdf':
            return self._load_pdf(file_path)
        else:
            logger.warning(f"Unsupported format: {extension}")
            return []
    
    def _load_txt(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load text file."""
        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                content = f.read()
            
            return [{
                'content': content,
                'metadata': {
                    'source': str(file_path),
                    'filename': file_path.name,
                    'format': 'txt'
                }
            }]
        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {e}")
            return []
    
    def _load_json(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load JSON file."""
        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                documents = []
                for item in data:
                    if isinstance(item, dict) and 'content' in item:
                        documents.append(item)
                    elif isinstance(item, str):
                        documents.append({
                            'content': item,
                            'metadata': {'source': str(file_path), 'filename': file_path.name}
                        })
                return documents
            elif isinstance(data, dict):
                if 'content' in data:
                    return [data]
                else:
                    # Treat the entire dict as content
                    return [{
                        'content': json.dumps(data),
                        'metadata': {'source': str(file_path), 'filename': file_path.name}
                    }]
            else:
                return [{
                    'content': str(data),
                    'metadata': {'source': str(file_path), 'filename': file_path.name}
                }]
        except Exception as e:
            logger.error(f"Error reading JSON file {file_path}: {e}")
            return []
    
    def _load_csv(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load CSV file."""
        try:
            documents = []
            with open(file_path, 'r', encoding=self.encoding) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Try to find a content column
                    content = None
                    for key in ['content', 'text', 'description', 'body']:
                        if key in row:
                            content = row[key]
                            break
                    
                    if content is None:
                        # Concatenate all columns
                        content = ' '.join(str(v) for v in row.values())
                    
                    documents.append({
                        'content': content,
                        'metadata': {
                            'source': str(file_path),
                            'filename': file_path.name,
                            'format': 'csv',
                            **row
                        }
                    })
            
            return documents
        except Exception as e:
            logger.error(f"Error reading CSV file {file_path}: {e}")
            return []
    
    def _load_pdf(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load PDF file (basic implementation)."""
        # Note: PDF parsing requires additional libraries like PyPDF2 or pdfplumber
        # This is a placeholder implementation
        logger.warning(f"PDF parsing not fully implemented. Skipping {file_path}")
        return [{
            'content': f"[PDF content from {file_path.name}]",
            'metadata': {
                'source': str(file_path),
                'filename': file_path.name,
                'format': 'pdf',
                'note': 'PDF parsing requires additional libraries'
            }
        }]
    
    def _is_supported_format(self, file_path: Path) -> bool:
        """Check if file format is supported."""
        extension = file_path.suffix.lower().lstrip('.')
        return extension in self.supported_formats
    
    def _load_huggingface_dataset(self, hf_path: str) -> List[Dict[str, Any]]:
        """
        Load data from Hugging Face dataset.
        
        Args:
            hf_path: Hugging Face dataset path (e.g., hf://datasets/...)
            
        Returns:
            List of document dictionaries
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required to load Hugging Face datasets. Install it with: pip install pandas")
        
        try:
            logger.info(f"Loading Hugging Face dataset from {hf_path}")
            
            # Load JSONL data
            df = pd.read_json(hf_path, lines=True)
            
            # Special handling for Australian Legal Corpus
            if "open-australian-legal-corpus" in hf_path:
                documents = self._convert_legal_corpus_to_documents(df)
            else:
                # Generic JSONL loading
                documents = self._convert_dataframe_to_documents(df)
            
            logger.info(f"Successfully loaded {len(documents)} documents from Hugging Face dataset")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading Hugging Face dataset: {e}")
            raise
    
    def _convert_legal_corpus_to_documents(self, df: 'pd.DataFrame') -> List[Dict[str, Any]]:
        """
        Convert Australian Legal Corpus DataFrame to document format.
        
        Args:
            df: pandas DataFrame with legal corpus data
            
        Returns:
            List of document dictionaries
        """
        documents = []
        
        for idx, row in df.iterrows():
            # Extract content from the row
            # Typical fields might include: text, title, citation, jurisdiction, etc.
            content = ""
            metadata = {
                'doc_id': f"legal_doc_{idx}",
                'source': 'open-australian-legal-corpus'
            }
            
            # Try to find the main text field
            text_fields = ['text', 'content', 'body', 'decision_text', 'full_text']
            for field in text_fields:
                if field in row and pd.notna(row[field]):
                    content = str(row[field])
                    break
            
            # If no text field found, concatenate all string fields
            if not content:
                content = " ".join(str(val) for val in row.values if pd.notna(val) and isinstance(val, str))
            
            # Add all other fields as metadata
            for col in df.columns:
                if col not in text_fields and pd.notna(row[col]):
                    metadata[col] = row[col]
            
            if content.strip():
                documents.append({
                    'content': content,
                    'metadata': metadata
                })
        
        return documents
    
    def _convert_dataframe_to_documents(self, df: 'pd.DataFrame') -> List[Dict[str, Any]]:
        """
        Convert a generic pandas DataFrame to document format.
        
        Args:
            df: pandas DataFrame
            
        Returns:
            List of document dictionaries
        """
        documents = []
        
        # Try to identify content columns
        content_columns = ['content', 'text', 'body', 'description', 'message']
        content_col = None
        
        for col in content_columns:
            if col in df.columns:
                content_col = col
                break
        
        for idx, row in df.iterrows():
            if content_col:
                content = str(row[content_col]) if pd.notna(row[content_col]) else ""
                metadata = {k: v for k, v in row.items() if k != content_col and pd.notna(v)}
            else:
                # Use all columns as content
                content = " ".join(str(val) for val in row.values if pd.notna(val))
                metadata = {}
            
            metadata['doc_id'] = metadata.get('id', f"doc_{idx}")
            metadata['source'] = 'dataframe'
            
            if content.strip():
                documents.append({
                    'content': content,
                    'metadata': metadata
                })
        
        return documents
    
    def load_australian_legal_corpus(self) -> List[Dict[str, Any]]:
        """
        Convenience method to load the Australian Legal Corpus dataset.
        
        Returns:
            List of document dictionaries
        """
        hf_path = "hf://datasets/isaacus/open-australian-legal-corpus/corpus.jsonl"
        return self._load_huggingface_dataset(hf_path)
    
    def validate_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate document structure and content.
        
        Args:
            documents: List of documents to validate
            
        Returns:
            List of valid documents
        """
        valid_documents = []
        
        for i, doc in enumerate(documents):
            if not isinstance(doc, dict):
                logger.warning(f"Document {i} is not a dictionary. Skipping.")
                continue
            
            if 'content' not in doc:
                logger.warning(f"Document {i} missing 'content' field. Skipping.")
                continue
            
            if not doc['content'] or not isinstance(doc['content'], str):
                logger.warning(f"Document {i} has invalid content. Skipping.")
                continue
            
            # Ensure metadata exists
            if 'metadata' not in doc:
                doc['metadata'] = {}
            
            doc['metadata']['doc_id'] = doc['metadata'].get('doc_id', f"doc_{i}")
            
            valid_documents.append(doc)
        
        logger.info(f"Validated {len(valid_documents)} out of {len(documents)} documents")
        return valid_documents
