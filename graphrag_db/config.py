"""
Configuration management for GraphRAG database system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
import os


@dataclass
class Config:
    """Configuration class for GraphRAG database setup."""
    
    # Data preprocessing settings
    input_data_path: str = "data/sample"
    supported_formats: List[str] = field(default_factory=lambda: ["txt", "json", "csv", "pdf"])
    encoding: str = "utf-8"
    
    # Data cleaning settings
    remove_html: bool = True
    remove_urls: bool = True
    remove_special_chars: bool = False
    normalize_whitespace: bool = True
    lowercase: bool = False
    remove_duplicates: bool = True
    min_text_length: int = 10
    
    # Chunking settings
    chunk_size: int = 512
    chunk_overlap: int = 50
    chunking_strategy: str = "fixed"  # Options: "fixed", "sentence", "paragraph"
    preserve_sentences: bool = True
    
    # Embedding settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    batch_size: int = 32
    device: str = "cpu"  # Options: "cpu", "cuda", "mps"
    normalize_embeddings: bool = True
    
    # Metadata settings
    extract_entities: bool = True
    extract_keywords: bool = True
    extract_summary: bool = False
    max_keywords: int = 10
    
    # Database settings
    vector_db_type: str = "chromadb"  # Options: "chromadb", "faiss", "qdrant"
    db_path: str = "data/vector_db"
    collection_name: str = "graphrag_collection"
    distance_metric: str = "cosine"  # Options: "cosine", "euclidean", "dot"
    
    # Pipeline settings
    num_workers: int = 4
    enable_logging: bool = True
    log_level: str = "INFO"
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
        }
    
    def save(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'Config':
        """Load configuration from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'Config':
        """Create config from dictionary."""
        return cls(**config_dict)
