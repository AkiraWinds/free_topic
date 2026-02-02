"""
Pipeline module for orchestrating the GraphRAG database setup process.
"""

import logging
from typing import List, Dict, Any, Optional
import time

from .config import Config
from .data_collection import DataPreprocessor
from .data_preprocessing import DataCleaner
from .chunking.chunking_part import TextChunker
from .embedding import EmbeddingGenerator
from .metadata_manager import MetadataManager
from .database_manager import DatabaseManager

logger = logging.getLogger(__name__)


class GraphRAGPipeline:
    """Orchestrates the complete GraphRAG database setup pipeline."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or Config()
        self._setup_logging()
        
        # Initialize components
        logger.info("Initializing GraphRAG pipeline components")
        self.preprocessor = DataPreprocessor(self.config)
        self.cleaner = DataCleaner(self.config)
        self.chunker = TextChunker(self.config)
        self.embedding_generator = EmbeddingGenerator(self.config)
        self.metadata_manager = MetadataManager(self.config)
        self.database = DatabaseManager(self.config)
        
        logger.info("GraphRAG pipeline initialized successfully")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        if self.config.enable_logging:
            log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
            logging.basicConfig(
                level=log_level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    def run(self, input_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete GraphRAG pipeline.
        
        Args:
            input_path: Path to input data. If None, uses config default.
            
        Returns:
            Dictionary with pipeline statistics and results
        """
        logger.info("=" * 80)
        logger.info("Starting GraphRAG Pipeline Execution")
        logger.info("=" * 80)
        
        start_time = time.time()
        stats = {
            'success': False,
            'stages': {}
        }
        
        try:
            # Stage 1: Data Preprocessing
            logger.info("\n[Stage 1/6] Data Preprocessing")
            logger.info("-" * 40)
            documents = self.preprocessor.load_data(input_path)
            documents = self.preprocessor.validate_documents(documents)
            stats['stages']['preprocessing'] = {
                'documents_loaded': len(documents),
                'status': 'completed'
            }
            logger.info(f"✓ Loaded and validated {len(documents)} documents")
            
            if not documents:
                logger.error("No documents loaded. Pipeline cannot continue.")
                stats['error'] = "No documents loaded"
                return stats
            
            # Stage 2: Data Cleaning
            logger.info("\n[Stage 2/6] Data Cleaning")
            logger.info("-" * 40)
            cleaned_documents = self.cleaner.clean_documents(documents)
            stats['stages']['cleaning'] = {
                'documents_cleaned': len(cleaned_documents),
                'documents_filtered': len(documents) - len(cleaned_documents),
                'status': 'completed'
            }
            logger.info(f"✓ Cleaned {len(cleaned_documents)} documents")
            
            if not cleaned_documents:
                logger.error("No documents remaining after cleaning. Pipeline cannot continue.")
                stats['error'] = "All documents filtered during cleaning"
                return stats
            
            # Stage 3: Text Chunking
            logger.info("\n[Stage 3/6] Text Chunking")
            logger.info("-" * 40)
            chunks = self.chunker.chunk_documents(cleaned_documents)
            chunk_stats = self.chunker.get_chunk_stats(chunks)
            stats['stages']['chunking'] = {
                **chunk_stats,
                'status': 'completed'
            }
            logger.info(f"✓ Created {len(chunks)} chunks")
            logger.info(f"  - Average chunk size: {chunk_stats.get('avg_chunk_size', 0):.0f} characters")
            
            if not chunks:
                logger.error("No chunks created. Pipeline cannot continue.")
                stats['error'] = "No chunks created"
                return stats
            
            # Stage 4: Metadata Enrichment
            logger.info("\n[Stage 4/6] Metadata Enrichment")
            logger.info("-" * 40)
            enriched_chunks = self.metadata_manager.enrich_metadata(chunks)
            metadata_stats = self.metadata_manager.get_metadata_stats(enriched_chunks)
            stats['stages']['metadata'] = {
                **metadata_stats,
                'status': 'completed'
            }
            logger.info(f"✓ Enriched metadata for {len(enriched_chunks)} chunks")
            logger.info(f"  - Metadata fields: {', '.join(metadata_stats.get('metadata_fields', []))}")
            
            # Stage 5: Embedding Generation
            logger.info("\n[Stage 5/6] Embedding Generation")
            logger.info("-" * 40)
            embedded_chunks = self.embedding_generator.generate_embeddings(enriched_chunks)
            embedding_info = self.embedding_generator.get_embedding_info()
            stats['stages']['embedding'] = {
                'chunks_embedded': len(embedded_chunks),
                'embedding_model': embedding_info.get('model_name', 'unknown'),
                'embedding_dimension': embedding_info.get('embedding_dimension', 0),
                'status': 'completed'
            }
            logger.info(f"✓ Generated embeddings for {len(embedded_chunks)} chunks")
            logger.info(f"  - Model: {embedding_info.get('model_name', 'unknown')}")
            logger.info(f"  - Dimension: {embedding_info.get('embedding_dimension', 0)}")
            
            # Stage 6: Database Storage
            logger.info("\n[Stage 6/6] Database Storage")
            logger.info("-" * 40)
            self.database.add_chunks(embedded_chunks)
            self.database.save()
            db_stats = self.database.get_stats()
            stats['stages']['database'] = {
                **db_stats,
                'status': 'completed'
            }
            logger.info(f"✓ Stored {len(embedded_chunks)} chunks in {db_stats.get('db_type', 'unknown')} database")
            
            # Final statistics
            end_time = time.time()
            stats['success'] = True
            stats['total_time_seconds'] = end_time - start_time
            stats['documents_processed'] = len(documents)
            stats['chunks_created'] = len(chunks)
            stats['chunks_stored'] = len(embedded_chunks)
            
            logger.info("\n" + "=" * 80)
            logger.info("Pipeline Execution Completed Successfully!")
            logger.info("=" * 80)
            logger.info(f"Total Time: {stats['total_time_seconds']:.2f} seconds")
            logger.info(f"Documents Processed: {stats['documents_processed']}")
            logger.info(f"Chunks Created: {stats['chunks_created']}")
            logger.info(f"Chunks Stored: {stats['chunks_stored']}")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"\n✗ Pipeline failed: {e}", exc_info=True)
            stats['success'] = False
            stats['error'] = str(e)
        
        return stats
    
    def query(self, query_text: str, top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Query the GraphRAG database.
        
        Args:
            query_text: Query text
            top_k: Number of results to return
            filters: Optional filters for the query
            
        Returns:
            List of relevant chunks with metadata
        """
        logger.info(f"Querying database: '{query_text}'")
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_query_embedding(query_text)
        
        # Query database
        results = self.database.query(query_embedding, top_k, filters)
        
        logger.info(f"Found {len(results)} results")
        return results
    
    def query_graph(self, entity: str, relationship_type: Optional[str] = None, depth: int = 2) -> Dict[str, Any]:
        """
        Query the knowledge graph for entity relationships.
        
        Args:
            entity: Entity to search for
            relationship_type: Optional relationship type filter
            depth: Graph traversal depth
            
        Returns:
            Graph structure with nodes and edges
        """
        logger.info(f"Querying graph for entity: '{entity}'")
        return self.database.query_graph(entity, relationship_type, depth)
    
    def load_australian_legal_corpus(self) -> Dict[str, Any]:
        """
        Convenience method to load and process the Australian Legal Corpus.
        
        Returns:
            Pipeline execution statistics
        """
        logger.info("Loading Australian Legal Corpus from Hugging Face")
        hf_path = "hf://datasets/isaacus/open-australian-legal-corpus/corpus.jsonl"
        return self.run(hf_path)
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get current database statistics."""
        return self.database.get_stats()
    
    def close(self) -> None:
        """Close database connections and cleanup resources."""
        logger.info("Closing pipeline resources")
        self.database.close()
        logger.info("Pipeline closed successfully")
