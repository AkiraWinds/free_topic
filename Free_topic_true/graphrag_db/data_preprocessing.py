"""
Data cleaning module for text normalization and noise removal.
"""

import re
from typing import List, Dict, Any, Set
import logging
from html import unescape

logger = logging.getLogger(__name__)


class DataCleaner:
    """Handles text cleaning, normalization, and deduplication."""
    
    def __init__(self, config):
        """
        Initialize the data cleaner.
        
        Args:
            config: Configuration object containing cleaning settings
        """
        self.config = config
        self.seen_hashes: Set[int] = set()
        
    def clean_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean a list of documents.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of cleaned documents
        """
        cleaned_documents = []
        
        for doc in documents:
            cleaned_doc = self.clean_document(doc)
            
            # Skip if content is too short after cleaning
            if len(cleaned_doc['content']) < self.config.min_text_length:
                logger.debug(f"Skipping document {doc.get('metadata', {}).get('doc_id', 'unknown')}: too short after cleaning")
                continue
            
            # Check for duplicates if enabled
            if self.config.remove_duplicates:
                content_hash = hash(cleaned_doc['content'])
                if content_hash in self.seen_hashes:
                    logger.debug(f"Skipping duplicate document")
                    continue
                self.seen_hashes.add(content_hash)
            
            cleaned_documents.append(cleaned_doc)
        
        logger.info(f"Cleaned {len(cleaned_documents)} out of {len(documents)} documents")
        return cleaned_documents
    
    def clean_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean a single document.
        
        Args:
            document: Document dictionary with 'content' and 'metadata'
            
        Returns:
            Cleaned document dictionary
        """
        content = document['content']
        
        # Remove HTML tags and entities
        if self.config.remove_html:
            content = self._remove_html(content)
        
        # Remove URLs
        if self.config.remove_urls:
            content = self._remove_urls(content)
        
        # Remove special characters
        if self.config.remove_special_chars:
            content = self._remove_special_chars(content)
        
        # Normalize whitespace
        if self.config.normalize_whitespace:
            content = self._normalize_whitespace(content)
        
        # Convert to lowercase
        if self.config.lowercase:
            content = content.lower()
        
        # Trim whitespace
        content = content.strip()
        
        return {
            'content': content,
            'metadata': document.get('metadata', {})
        }
    
    def _remove_html(self, text: str) -> str:
        """Remove HTML tags and decode HTML entities."""
        # Decode HTML entities
        text = unescape(text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        return text
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        # Remove http/https URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove www URLs
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        return text
    
    def _remove_special_chars(self, text: str) -> str:
        """Remove special characters, keeping only alphanumeric and basic punctuation."""
        # Keep letters, numbers, spaces, and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?;:\-\'"()\[\]{}]', '', text)
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace: replace multiple spaces/newlines with single space."""
        # Replace multiple whitespace characters with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace from lines
        text = '\n'.join(line.strip() for line in text.split('\n'))
        
        return text
    
    def reset_duplicates(self) -> None:
        """Reset the duplicate detection cache."""
        self.seen_hashes.clear()
        logger.info("Duplicate detection cache reset")
    
    def get_cleaning_stats(self) -> Dict[str, Any]:
        """Get statistics about the cleaning process."""
        return {
            'unique_documents_seen': len(self.seen_hashes),
            'remove_html': self.config.remove_html,
            'remove_urls': self.config.remove_urls,
            'remove_special_chars': self.config.remove_special_chars,
            'normalize_whitespace': self.config.normalize_whitespace,
            'lowercase': self.config.lowercase,
            'min_text_length': self.config.min_text_length
        }
