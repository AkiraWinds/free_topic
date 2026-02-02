"""
Metadata management module for extracting and managing document metadata.
"""

import re
from typing import List, Dict, Any, Set
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class MetadataManager:
    """Handles metadata extraction, enrichment, and management."""
    
    def __init__(self, config):
        """
        Initialize the metadata manager.
        
        Args:
            config: Configuration object containing metadata settings
        """
        self.config = config
        self.extract_entities = config.extract_entities
        self.extract_keywords = config.extract_keywords
        self.extract_summary = config.extract_summary
        self.max_keywords = config.max_keywords
        
    def enrich_metadata(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich chunks with additional metadata.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of chunks with enriched metadata
        """
        for chunk in chunks:
            self._enrich_single_chunk(chunk)
        
        logger.info(f"Enriched metadata for {len(chunks)} chunks")
        return chunks
    
    def _enrich_single_chunk(self, chunk: Dict[str, Any]) -> None:
        """Enrich a single chunk with metadata."""
        content = chunk['content']
        metadata = chunk.get('metadata', {})
        
        # Add basic statistics
        metadata['char_count'] = len(content)
        metadata['word_count'] = len(content.split())
        metadata['sentence_count'] = len(re.split(r'[.!?]+', content))
        
        # Extract entities if enabled
        if self.extract_entities:
            metadata['entities'] = self._extract_entities(content)
        
        # Extract keywords if enabled
        if self.extract_keywords:
            metadata['keywords'] = self._extract_keywords(content)
        
        # Generate summary if enabled
        if self.extract_summary:
            metadata['summary'] = self._generate_summary(content)
        
        chunk['metadata'] = metadata
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text.
        This is a simple pattern-based approach. For production, use NLP libraries like spaCy.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Dictionary of entity types and their values
        """
        entities = {
            'persons': [],
            'organizations': [],
            'locations': [],
            'dates': [],
            'emails': [],
            'urls': []
        }
        
        # Extract emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        entities['emails'] = list(set(re.findall(email_pattern, text)))
        
        # Extract URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        entities['urls'] = list(set(re.findall(url_pattern, text)))
        
        # Extract dates (simple patterns)
        date_patterns = [
            r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b'  # Month DD, YYYY
        ]
        dates = []
        for pattern in date_patterns:
            dates.extend(re.findall(pattern, text, re.IGNORECASE))
        entities['dates'] = list(set(dates))
        
        # Extract capitalized words as potential entities (simple heuristic)
        # This would be replaced with proper NER in production
        capitalized_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Filter out common words and sentence starts
        common_words = {'The', 'This', 'That', 'These', 'Those', 'A', 'An', 'In', 'On', 'At', 'To', 'For'}
        potential_entities = [w for w in capitalized_words if w not in common_words]
        
        # Simple heuristic: words appearing multiple times might be important entities
        entity_counter = Counter(potential_entities)
        important_entities = [entity for entity, count in entity_counter.items() if count > 1]
        
        entities['persons'] = important_entities[:5]  # Top 5 as persons (placeholder)
        
        return entities
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text using simple frequency-based approach.
        For production, use TF-IDF or more sophisticated methods.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of keywords
        """
        # Convert to lowercase and split into words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Remove common stop words
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'when', 'where', 'who', 'which', 'why', 'how'
        }
        
        # Filter words: remove stop words and short words
        filtered_words = [
            word for word in words
            if word not in stop_words and len(word) > 3
        ]
        
        # Count word frequencies
        word_counts = Counter(filtered_words)
        
        # Get top keywords
        keywords = [word for word, count in word_counts.most_common(self.max_keywords)]
        
        return keywords
    
    def _generate_summary(self, text: str, max_length: int = 200) -> str:
        """
        Generate a simple summary of the text.
        This is a very basic implementation. For production, use summarization models.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            
        Returns:
            Summary text
        """
        # Simple approach: take the first few sentences
        sentences = re.split(r'[.!?]+', text)
        
        summary = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if len(summary) + len(sentence) < max_length:
                summary += sentence + ". "
            else:
                break
        
        return summary.strip()
    
    def extract_metadata_field(self, chunks: List[Dict[str, Any]], field: str) -> List[Any]:
        """
        Extract a specific metadata field from all chunks.
        
        Args:
            chunks: List of chunk dictionaries
            field: Metadata field to extract
            
        Returns:
            List of field values
        """
        values = []
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            if field in metadata:
                values.append(metadata[field])
        
        return values
    
    def filter_by_metadata(self, chunks: List[Dict[str, Any]], 
                          filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Filter chunks by metadata criteria.
        
        Args:
            chunks: List of chunk dictionaries
            filters: Dictionary of field:value pairs to filter by
            
        Returns:
            Filtered list of chunks
        """
        filtered_chunks = []
        
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            matches = True
            
            for field, value in filters.items():
                if field not in metadata or metadata[field] != value:
                    matches = False
                    break
            
            if matches:
                filtered_chunks.append(chunk)
        
        logger.info(f"Filtered {len(filtered_chunks)} chunks out of {len(chunks)} using criteria: {filters}")
        return filtered_chunks
    
    def get_metadata_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about metadata across all chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Dictionary with metadata statistics
        """
        if not chunks:
            return {}
        
        stats = {
            'total_chunks': len(chunks),
            'metadata_fields': set()
        }
        
        # Collect all metadata fields
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            stats['metadata_fields'].update(metadata.keys())
        
        stats['metadata_fields'] = list(stats['metadata_fields'])
        
        # Calculate average word count if available
        word_counts = [
            chunk.get('metadata', {}).get('word_count', 0)
            for chunk in chunks
        ]
        if word_counts:
            stats['avg_word_count'] = sum(word_counts) / len(word_counts)
        
        return stats
