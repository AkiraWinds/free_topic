"""
Embedding generation module for converting text to vector representations.
"""

import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Handles text embedding generation with support for multiple models."""
    
    def __init__(self, config):
        """
        Initialize the embedding generator.
        
        Args:
            config: Configuration object containing embedding settings
        """
        self.config = config
        self.model_name = config.embedding_model
        self.embedding_dimension = config.embedding_dimension
        self.batch_size = config.batch_size
        self.device = config.device
        self.normalize = config.normalize_embeddings
        self.model = None
        self._load_model()
        
    def _load_model(self) -> None:
        """Load the embedding model."""
        try:
            # Try to import sentence-transformers
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Model loaded successfully on device: {self.device}")
            
        except ImportError:
            logger.warning("sentence-transformers not installed. Using fallback embedding.")
            self.model = None
        except Exception as e:
            logger.error(f"Error loading model: {e}. Using fallback embedding.")
            self.model = None
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'content'
            
        Returns:
            List of chunk dictionaries with added 'embedding' field
        """
        if not chunks:
            return []
        
        texts = [chunk['content'] for chunk in chunks]
        
        if self.model is not None:
            embeddings = self._generate_with_model(texts)
        else:
            embeddings = self._generate_fallback_embeddings(texts)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding
        
        logger.info(f"Generated embeddings for {len(chunks)} chunks")
        return chunks
    
    def _generate_with_model(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings using the loaded model."""
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            try:
                batch_embeddings = self.model.encode(
                    batch_texts,
                    normalize_embeddings=self.normalize,
                    show_progress_bar=False
                )
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i}: {e}")
                # Use fallback for failed batch
                fallback_embeddings = self._generate_fallback_embeddings(batch_texts)
                all_embeddings.extend(fallback_embeddings)
        
        return all_embeddings
    
    def _generate_fallback_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate simple fallback embeddings based on text statistics.
        This is a basic implementation for when proper embedding models are unavailable.
        """
        embeddings = []
        
        for text in texts:
            # Create a simple embedding based on text characteristics
            embedding = np.zeros(self.embedding_dimension)
            
            if text:
                # Use hash-based features for basic embedding
                for i, char in enumerate(text[:self.embedding_dimension]):
                    embedding[i % self.embedding_dimension] += ord(char) / 1000.0
                
                # Add some statistical features
                embedding[0] = len(text) / 1000.0
                embedding[1] = len(text.split()) / 100.0
                embedding[2] = text.count('.') / 10.0
                
                # Normalize if required
                if self.normalize:
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
            
            embeddings.append(embedding)
        
        return embeddings
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query text.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        if self.model is not None:
            try:
                embedding = self.model.encode(
                    query,
                    normalize_embeddings=self.normalize,
                    show_progress_bar=False
                )
                return embedding
            except Exception as e:
                logger.error(f"Error generating query embedding: {e}")
                return self._generate_fallback_embeddings([query])[0]
        else:
            return self._generate_fallback_embeddings([query])[0]
    
    def get_embedding_info(self) -> Dict[str, Any]:
        """Get information about the embedding configuration."""
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dimension,
            'batch_size': self.batch_size,
            'device': self.device,
            'normalize': self.normalize,
            'model_loaded': self.model is not None
        }
