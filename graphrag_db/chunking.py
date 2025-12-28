"""
Text chunking module for splitting documents using LangChain text splitters.
"""

import re
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

try:
    from langchain.text_splitter import (
        RecursiveCharacterTextSplitter,
        CharacterTextSplitter,
        TokenTextSplitter,
        SentenceTransformersTokenTextSplitter
    )
    from langchain.schema import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("langchain not available. Using fallback chunking methods.")


class TextChunker:
    """Handles text chunking with various strategies using LangChain text splitters."""
    
    def __init__(self, config):
        """
        Initialize the text chunker.
        
        Args:
            config: Configuration object containing chunking settings
        """
        self.config = config
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        self.strategy = config.chunking_strategy
        self.preserve_sentences = config.preserve_sentences
        self._initialize_splitter()
        
    def _initialize_splitter(self) -> None:
        """Initialize the LangChain text splitter based on strategy."""
        if not LANGCHAIN_AVAILABLE:
            self.splitter = None
            logger.warning("LangChain not available. Will use fallback chunking.")
            return
        
        try:
            if self.strategy == "fixed":
                # RecursiveCharacterTextSplitter tries to split on natural boundaries
                self.splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    length_function=len,
                    separators=["\n\n", "\n", ". ", " ", ""] if self.preserve_sentences else None
                )
            elif self.strategy == "sentence":
                # Use character splitter with sentence separators
                self.splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separators=["\n\n", "\n", ".", "!", "?", " "]
                )
            elif self.strategy == "paragraph":
                # Split by paragraphs
                self.splitter = CharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separator="\n\n"
                )
            elif self.strategy == "token":
                # Token-based splitting
                self.splitter = TokenTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
            else:
                logger.warning(f"Unknown strategy: {self.strategy}. Using RecursiveCharacterTextSplitter.")
                self.splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
            
            logger.info(f"Initialized LangChain text splitter: {type(self.splitter).__name__}")
        except Exception as e:
            logger.error(f"Error initializing LangChain splitter: {e}")
            self.splitter = None
        
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk a list of documents using LangChain text splitters.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of chunked document dictionaries
        """
        all_chunks = []
        
        for doc in documents:
            if LANGCHAIN_AVAILABLE and self.splitter is not None:
                chunks = self._chunk_document_with_langchain(doc)
            else:
                chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents using LangChain")
        return all_chunks
    
    def _chunk_document_with_langchain(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a single document using LangChain text splitters.
        
        Args:
            document: Document dictionary with 'content' and 'metadata'
            
        Returns:
            List of chunk dictionaries
        """
        content = document['content']
        metadata = document.get('metadata', {})
        
        try:
            # Create LangChain Document
            langchain_doc = Document(page_content=content, metadata=metadata)
            
            # Split using the configured splitter
            split_docs = self.splitter.split_documents([langchain_doc])
            
            # Convert back to our format
            chunk_documents = []
            for i, split_doc in enumerate(split_docs):
                chunk_doc = {
                    'content': split_doc.page_content,
                    'metadata': {
                        **split_doc.metadata,
                        'chunk_id': f"{metadata.get('doc_id', 'unknown')}_chunk_{i}",
                        'chunk_index': i,
                        'total_chunks': len(split_docs),
                        'chunk_size': len(split_doc.page_content)
                    }
                }
                chunk_documents.append(chunk_doc)
            
            return chunk_documents
            
        except Exception as e:
            logger.error(f"Error chunking with LangChain: {e}. Using fallback.")
            return self.chunk_document(document)
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a single document based on the configured strategy.
        
        Args:
            document: Document dictionary with 'content' and 'metadata'
            
        Returns:
            List of chunk dictionaries
        """
        content = document['content']
        metadata = document.get('metadata', {})
        
        if self.strategy == "fixed":
            chunks = self._chunk_fixed_size(content)
        elif self.strategy == "sentence":
            chunks = self._chunk_by_sentences(content)
        elif self.strategy == "paragraph":
            chunks = self._chunk_by_paragraphs(content)
        else:
            logger.warning(f"Unknown chunking strategy: {self.strategy}. Using fixed size.")
            chunks = self._chunk_fixed_size(content)
        
        # Create chunk documents with metadata
        chunk_documents = []
        for i, chunk_text in enumerate(chunks):
            chunk_doc = {
                'content': chunk_text,
                'metadata': {
                    **metadata,
                    'chunk_id': f"{metadata.get('doc_id', 'unknown')}_chunk_{i}",
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'chunk_size': len(chunk_text)
                }
            }
            chunk_documents.append(chunk_doc)
        
        return chunk_documents
    
    def _chunk_fixed_size(self, text: str) -> List[str]:
        """
        Chunk text into fixed-size pieces with overlap.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # If preserve_sentences is enabled, try to break at sentence boundary
            if self.preserve_sentences and end < len(text):
                # Look for sentence ending within a small window
                search_start = max(start, end - 100)
                search_text = text[search_start:end + 100]
                
                # Find sentence boundaries (. ! ? followed by space or end)
                sentence_endings = [m.end() for m in re.finditer(r'[.!?]\s+', search_text)]
                
                if sentence_endings:
                    # Find the closest sentence ending to our target
                    target_pos = end - search_start
                    closest_ending = min(sentence_endings, key=lambda x: abs(x - target_pos))
                    end = search_start + closest_ending
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            
            # Prevent infinite loop - if we're not making progress, break
            if start >= end or (len(chunks) > 0 and start >= len(text)):
                break
        
        return chunks
    
    def _chunk_by_sentences(self, text: str) -> List[str]:
        """
        Chunk text by sentences, respecting max chunk size.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence exceeds chunk size, save current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap (last few sentences)
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s)
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_length
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _chunk_by_paragraphs(self, text: str) -> List[str]:
        """
        Chunk text by paragraphs, respecting max chunk size.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        # Split into paragraphs (double newline or more)
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            paragraph_length = len(paragraph)
            
            # If single paragraph exceeds chunk size, split it
            if paragraph_length > self.chunk_size:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split large paragraph using fixed size chunking
                sub_chunks = self._chunk_fixed_size(paragraph)
                chunks.extend(sub_chunks)
                continue
            
            # If adding this paragraph exceeds chunk size, save current chunk
            if current_length + paragraph_length > self.chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [paragraph]
                current_length = paragraph_length
            else:
                current_chunk.append(paragraph)
                current_length += paragraph_length
        
        # Add the last chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def get_chunk_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Dictionary with statistics
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'avg_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0
            }
        
        chunk_sizes = [len(chunk['content']) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'strategy': self.strategy,
            'configured_chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap
        }
