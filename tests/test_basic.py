"""
Basic tests for GraphRAG database system.
"""

import os
import sys
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graphrag_db import Config, GraphRAGPipeline


class TestConfig:
    """Test configuration management."""
    
    def test_config_creation(self):
        """Test creating a config with default values."""
        config = Config()
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50
        assert config.embedding_dimension == 384
    
    def test_config_custom_values(self):
        """Test creating a config with custom values."""
        config = Config(
            chunk_size=1024,
            chunk_overlap=100,
            vector_db_type="memory"
        )
        assert config.chunk_size == 1024
        assert config.chunk_overlap == 100
        assert config.vector_db_type == "memory"
    
    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = Config(chunk_size=256)
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict['chunk_size'] == 256


class TestPipeline:
    """Test pipeline functionality."""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        config = Config(vector_db_type="memory")
        pipeline = GraphRAGPipeline(config)
        assert pipeline is not None
        assert pipeline.config.vector_db_type == "memory"
        pipeline.close()
    
    def test_pipeline_with_sample_data(self):
        """Test running pipeline with sample data."""
        config = Config(
            input_data_path="data/sample",
            chunk_size=256,
            vector_db_type="memory",
            enable_logging=False
        )
        
        pipeline = GraphRAGPipeline(config)
        stats = pipeline.run()
        
        assert stats['success'] is True
        assert stats.get('documents_processed', 0) > 0
        assert stats.get('chunks_created', 0) > 0
        
        pipeline.close()
    
    def test_query_functionality(self):
        """Test querying the database."""
        config = Config(
            input_data_path="data/sample",
            chunk_size=256,
            vector_db_type="memory",
            enable_logging=False
        )
        
        pipeline = GraphRAGPipeline(config)
        stats = pipeline.run()
        
        if stats['success']:
            results = pipeline.query("test query", top_k=3)
            assert isinstance(results, list)
            assert len(results) <= 3
        
        pipeline.close()


class TestDataPreprocessing:
    """Test data preprocessing functionality."""
    
    def test_load_text_file(self):
        """Test loading a text file."""
        from graphrag_db.data_preprocessing import DataPreprocessor
        
        config = Config()
        preprocessor = DataPreprocessor(config)
        
        # Load sample data
        documents = preprocessor.load_data("data/sample/sample_doc1.txt")
        
        assert len(documents) > 0
        assert 'content' in documents[0]
        assert 'metadata' in documents[0]
    
    def test_validate_documents(self):
        """Test document validation."""
        from graphrag_db.data_preprocessing import DataPreprocessor
        
        config = Config()
        preprocessor = DataPreprocessor(config)
        
        # Valid documents
        valid_docs = [
            {'content': 'Test content', 'metadata': {}},
            {'content': 'Another test', 'metadata': {}}
        ]
        
        validated = preprocessor.validate_documents(valid_docs)
        assert len(validated) == 2
        
        # Invalid documents
        invalid_docs = [
            {'content': 'Valid'},
            {'no_content': 'Invalid'},
            {'content': ''}
        ]
        
        validated = preprocessor.validate_documents(invalid_docs)
        assert len(validated) == 1  # Only the first one is valid


class TestChunking:
    """Test chunking functionality."""
    
    def test_chunk_document(self):
        """Test chunking a single document."""
        from graphrag_db.chunking import TextChunker
        
        config = Config(chunk_size=100, chunk_overlap=20)
        chunker = TextChunker(config)
        
        doc = {
            'content': 'This is a test document. ' * 20,  # Long text
            'metadata': {'doc_id': 'test_doc'}
        }
        
        chunks = chunker.chunk_document(doc)
        
        assert len(chunks) > 1
        assert all('content' in chunk for chunk in chunks)
        assert all('metadata' in chunk for chunk in chunks)


class TestMetadata:
    """Test metadata management."""
    
    def test_extract_keywords(self):
        """Test keyword extraction."""
        from graphrag_db.metadata_manager import MetadataManager
        
        config = Config(extract_keywords=True, max_keywords=5)
        manager = MetadataManager(config)
        
        chunks = [{
            'content': 'Python programming language is great for data science and machine learning.',
            'metadata': {}
        }]
        
        enriched = manager.enrich_metadata(chunks)
        
        assert len(enriched) == 1
        assert 'keywords' in enriched[0]['metadata']
        assert isinstance(enriched[0]['metadata']['keywords'], list)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
