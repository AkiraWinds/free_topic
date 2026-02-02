"""
GraphRAG Database Setup Project
A comprehensive system for building knowledge graphs with RAG capabilities.
"""

__version__ = "0.1.0"

from .config import Config
from .data_collection import DataPreprocessor
from .data_preprocessing import DataCleaner
from .chunking.chunking_part import TextChunker
from .embedding import EmbeddingGenerator
from .metadata_manager import MetadataManager
from .database_manager import DatabaseManager
from .pipeline import GraphRAGPipeline

__all__ = [
    "Config",
    "DataPreprocessor",
    "DataCleaner",
    "TextChunker",
    "EmbeddingGenerator",
    "MetadataManager",
    "DatabaseManager",
    "GraphRAGPipeline",
]
