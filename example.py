"""
Example usage of the GraphRAG Database Setup System.

This script demonstrates how to:
1. Initialize the GraphRAG pipeline
2. Load data from various sources including Hugging Face
3. Process data through the pipeline
4. Query the database
5. Explore the knowledge graph
"""

import os
import logging
from graphrag_db import Config, GraphRAGPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_basic_usage():
    """Basic usage example with local files."""
    print("\n" + "=" * 80)
    print("Example 1: Basic Usage with Local Files")
    print("=" * 80 + "\n")
    
    # Create configuration
    config = Config(
        input_data_path="data/sample",
        chunk_size=512,
        chunk_overlap=50,
        chunking_strategy="fixed",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        vector_db_type="memory",  # Use in-memory for quick demo
        enable_logging=True
    )
    
    # Initialize pipeline
    pipeline = GraphRAGPipeline(config)
    
    # Run the pipeline
    stats = pipeline.run()
    
    # Print results
    print("\nPipeline Results:")
    print(f"  Success: {stats['success']}")
    print(f"  Total Time: {stats.get('total_time_seconds', 0):.2f} seconds")
    print(f"  Documents Processed: {stats.get('documents_processed', 0)}")
    print(f"  Chunks Created: {stats.get('chunks_created', 0)}")
    
    # Query example
    if stats['success'] and stats.get('chunks_stored', 0) > 0:
        print("\nQuerying the database...")
        results = pipeline.query("example query", top_k=3)
        print(f"Found {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"  Content preview: {result.get('content', '')[:100]}...")
    
    # Cleanup
    pipeline.close()


def example_australian_legal_corpus():
    """Example using the Australian Legal Corpus from Hugging Face."""
    print("\n" + "=" * 80)
    print("Example 2: Australian Legal Corpus from Hugging Face")
    print("=" * 80 + "\n")
    
    # Create configuration optimized for legal documents
    config = Config(
        chunk_size=1024,  # Larger chunks for legal text
        chunk_overlap=100,
        chunking_strategy="paragraph",  # Preserve legal paragraphs
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        vector_db_type="neo4j",  # Use Neo4j for graph capabilities
        extract_entities=True,
        extract_keywords=True,
        enable_logging=True,
        log_level="INFO"
    )
    
    # Initialize pipeline
    pipeline = GraphRAGPipeline(config)
    
    # Load and process Australian Legal Corpus
    print("Note: This requires Hugging Face authentication for the dataset.")
    print("Use: huggingface-cli login")
    print("\nProcessing Australian Legal Corpus...")
    
    try:
        stats = pipeline.load_australian_legal_corpus()
        
        # Print results
        print("\nPipeline Results:")
        print(f"  Success: {stats['success']}")
        if stats['success']:
            print(f"  Total Time: {stats.get('total_time_seconds', 0):.2f} seconds")
            print(f"  Documents Processed: {stats.get('documents_processed', 0)}")
            print(f"  Chunks Created: {stats.get('chunks_created', 0)}")
            
            # Example queries
            print("\n--- Example Queries ---")
            
            # Semantic search
            print("\n1. Semantic Search:")
            results = pipeline.query("contract dispute resolution", top_k=3)
            for i, result in enumerate(results, 1):
                print(f"\nResult {i}:")
                print(f"  Similarity: {result.get('similarity', 0):.4f}")
                print(f"  Content: {result.get('content', '')[:200]}...")
            
            # Graph query
            print("\n2. Knowledge Graph Query:")
            graph = pipeline.query_graph("Court", depth=2)
            print(f"  Nodes found: {len(graph.get('nodes', []))}")
            print(f"  Edges found: {len(graph.get('edges', []))}")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure you have access to the Hugging Face dataset.")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        pipeline.close()


def example_neo4j_setup():
    """Example showing Neo4j configuration."""
    print("\n" + "=" * 80)
    print("Example 3: Neo4j Graph Database Setup")
    print("=" * 80 + "\n")
    
    print("Neo4j Configuration:")
    print("  1. Install Neo4j: https://neo4j.com/download/")
    print("  2. Start Neo4j server")
    print("  3. Set environment variables:")
    print("     - NEO4J_URI=bolt://localhost:7687")
    print("     - NEO4J_USERNAME=neo4j")
    print("     - NEO4J_PASSWORD=your_password")
    print()
    
    # Check if Neo4j is configured
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    
    if uri and username and password:
        print("✓ Neo4j environment variables are set")
        print(f"  URI: {uri}")
        print(f"  Username: {username}")
        
        # Create configuration with Neo4j
        config = Config(
            vector_db_type="neo4j",
            chunk_size=512,
            extract_entities=True,
            extract_keywords=True
        )
        
        # Initialize pipeline
        try:
            pipeline = GraphRAGPipeline(config)
            stats = pipeline.get_database_stats()
            print(f"\n✓ Connected to Neo4j successfully")
            print(f"  Database type: {stats.get('db_type', 'unknown')}")
            print(f"  Total chunks: {stats.get('total_chunks', 0)}")
            print(f"  Total documents: {stats.get('total_documents', 0)}")
            print(f"  Total entities: {stats.get('total_entities', 0)}")
            pipeline.close()
        except Exception as e:
            print(f"\n✗ Failed to connect to Neo4j: {e}")
    else:
        print("✗ Neo4j environment variables are not set")
        print("  Set them using:")
        print("    export NEO4J_URI=bolt://localhost:7687")
        print("    export NEO4J_USERNAME=neo4j")
        print("    export NEO4J_PASSWORD=your_password")


def example_custom_config():
    """Example showing custom configuration."""
    print("\n" + "=" * 80)
    print("Example 4: Custom Configuration")
    print("=" * 80 + "\n")
    
    # Create custom configuration
    config = Config(
        # Data preprocessing
        input_data_path="data/sample",
        supported_formats=["txt", "json", "csv", "pdf"],
        encoding="utf-8",
        
        # Data cleaning
        remove_html=True,
        remove_urls=True,
        normalize_whitespace=True,
        remove_duplicates=True,
        min_text_length=50,
        
        # Chunking (using LangChain)
        chunk_size=768,
        chunk_overlap=100,
        chunking_strategy="sentence",  # Options: "fixed", "sentence", "paragraph", "token"
        preserve_sentences=True,
        
        # Embedding
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension=384,
        batch_size=32,
        device="cpu",
        normalize_embeddings=True,
        
        # Metadata
        extract_entities=True,
        extract_keywords=True,
        extract_summary=False,
        max_keywords=15,
        
        # Database (Neo4j)
        vector_db_type="neo4j",
        collection_name="my_custom_collection",
        distance_metric="cosine",
        
        # Pipeline
        enable_logging=True,
        log_level="INFO"
    )
    
    # Save configuration
    config.save("config/custom_config.json")
    print("✓ Configuration saved to config/custom_config.json")
    
    # Load configuration
    loaded_config = Config.load("config/custom_config.json")
    print("✓ Configuration loaded from file")
    print(f"  Chunk size: {loaded_config.chunk_size}")
    print(f"  Chunking strategy: {loaded_config.chunking_strategy}")
    print(f"  Database type: {loaded_config.vector_db_type}")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("GraphRAG Database Setup System - Examples")
    print("=" * 80)
    
    # Example 1: Basic usage
    try:
        example_basic_usage()
    except Exception as e:
        logger.error(f"Example 1 failed: {e}")
    
    # Example 2: Australian Legal Corpus (commented out by default)
    # Uncomment to run this example:
    # try:
    #     example_australian_legal_corpus()
    # except Exception as e:
    #     logger.error(f"Example 2 failed: {e}")
    
    # Example 3: Neo4j setup
    try:
        example_neo4j_setup()
    except Exception as e:
        logger.error(f"Example 3 failed: {e}")
    
    # Example 4: Custom configuration
    try:
        example_custom_config()
    except Exception as e:
        logger.error(f"Example 4 failed: {e}")
    
    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
