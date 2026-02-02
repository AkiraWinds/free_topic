# GraphRAG Database Setup - Usage Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Data Loading](#data-loading)
5. [Pipeline Execution](#pipeline-execution)
6. [Querying](#querying)
7. [Neo4j Integration](#neo4j-integration)
8. [Advanced Usage](#advanced-usage)

## Quick Start

### 1. Install Dependencies

```bash
# Core dependencies
pip install numpy pandas

# Optional: For full functionality
pip install langchain langchain-community sentence-transformers neo4j
```

### 2. Run with Sample Data

```python
from graphrag_db import Config, GraphRAGPipeline

# Initialize with default config
config = Config(
    input_data_path="data/sample",
    vector_db_type="memory"  # Use in-memory for quick start
)

# Run pipeline
pipeline = GraphRAGPipeline(config)
stats = pipeline.run()

# Query
results = pipeline.query("your query here", top_k=5)
for result in results:
    print(f"Content: {result['content'][:100]}...")
    print(f"Similarity: {result.get('similarity', 0):.4f}\n")

pipeline.close()
```

## Installation

### Minimal Installation (No External Dependencies)

```bash
pip install numpy pandas
```

This allows basic functionality with:
- In-memory storage
- Fallback embeddings
- Basic text splitting

### Full Installation

```bash
pip install -r requirements.txt
```

This includes:
- LangChain for advanced document loading and text splitting
- Sentence Transformers for quality embeddings
- Neo4j driver for graph database
- ChromaDB and FAISS as alternative vector stores

### Neo4j Setup

#### Option 1: Docker (Recommended)

```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  -v $HOME/neo4j/data:/data \
  neo4j:latest
```

#### Option 2: Desktop Application

1. Download from [neo4j.com/download](https://neo4j.com/download/)
2. Install and start Neo4j Desktop
3. Create a new project and database
4. Start the database

#### Environment Variables

Create a `.env` file:

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

Or export them:

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=your_password
```

## Configuration

### Basic Configuration

```python
from graphrag_db import Config

config = Config(
    # Input
    input_data_path="data/my_documents",
    
    # Chunking
    chunk_size=512,
    chunk_overlap=50,
    chunking_strategy="fixed",  # or "sentence", "paragraph", "token"
    
    # Database
    vector_db_type="neo4j",  # or "memory", "chromadb", "faiss"
    
    # Metadata
    extract_entities=True,
    extract_keywords=True
)
```

### Save/Load Configuration

```python
# Save
config.save("config/my_config.json")

# Load
config = Config.load("config/my_config.json")
```

## Data Loading

### Local Files

```python
# Single file
stats = pipeline.run("data/document.txt")

# Directory
stats = pipeline.run("data/my_documents/")
```

### Supported Formats

- Text files (.txt)
- JSON files (.json)
- CSV files (.csv)
- PDF files (.pdf) - requires PyPDF2

### Australian Legal Corpus (Hugging Face)

```python
from graphrag_db import Config, GraphRAGPipeline

config = Config(
    chunk_size=1024,
    vector_db_type="neo4j"
)

pipeline = GraphRAGPipeline(config)

# Login to Hugging Face first
# huggingface-cli login

# Load the corpus
stats = pipeline.load_australian_legal_corpus()
```

### Custom Data Loading

```python
from graphrag_db.data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor(config)

# Load with custom path
documents = preprocessor.load_data("hf://datasets/your-username/your-dataset/file.jsonl")
```

## Pipeline Execution

### Full Pipeline

```python
from graphrag_db import Config, GraphRAGPipeline

config = Config(input_data_path="data/sample")
pipeline = GraphRAGPipeline(config)

# Run all stages
stats = pipeline.run()

# Check results
if stats['success']:
    print(f"Processed {stats['documents_processed']} documents")
    print(f"Created {stats['chunks_created']} chunks")
    print(f"Time: {stats['total_time_seconds']:.2f}s")
```

### Pipeline Stages

The pipeline runs through 6 stages:

1. **Data Preprocessing**: Load and validate documents
2. **Data Cleaning**: Normalize and clean text
3. **Text Chunking**: Split into manageable pieces
4. **Metadata Enrichment**: Extract entities and keywords
5. **Embedding Generation**: Create vector embeddings
6. **Database Storage**: Store in graph database

### Stage-by-Stage Execution

```python
# Load data
documents = pipeline.preprocessor.load_data()

# Clean
cleaned = pipeline.cleaner.clean_documents(documents)

# Chunk
chunks = pipeline.chunker.chunk_documents(cleaned)

# Enrich
enriched = pipeline.metadata_manager.enrich_metadata(chunks)

# Embed
embedded = pipeline.embedding_generator.generate_embeddings(enriched)

# Store
pipeline.database.add_chunks(embedded)
```

## Querying

### Semantic Search

```python
# Basic query
results = pipeline.query("machine learning algorithms", top_k=5)

for result in results:
    print(f"Content: {result['content']}")
    print(f"Similarity: {result.get('similarity', 0):.4f}")
    print(f"Source: {result['metadata'].get('source', 'unknown')}")
    print()
```

### Filtered Search

```python
# Query with filters
results = pipeline.query(
    "contract law",
    top_k=10,
    filters={'doc_id': 'legal_doc_123'}
)
```

### Knowledge Graph Queries (Neo4j)

```python
# Explore entity relationships
graph = pipeline.query_graph(
    entity="Court",
    relationship_type="MENTIONS",
    depth=2
)

print(f"Nodes: {len(graph['nodes'])}")
print(f"Edges: {len(graph['edges'])}")

# Visualize in Neo4j Browser at http://localhost:7474
```

### Direct Cypher Queries (Neo4j)

If using Neo4j, you can run custom Cypher queries:

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password")
)

with driver.session() as session:
    # Find all chunks mentioning a specific entity
    result = session.run("""
        MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {value: $entity})
        RETURN c.content, c.chunk_id
        LIMIT 10
    """, entity="Contract")
    
    for record in result:
        print(record['c.content'])
```

## Neo4j Integration

### Graph Structure

When using Neo4j, the system creates:

**Nodes:**
- `Chunk`: Text chunks with embeddings
- `Document`: Source documents
- `Entity`: Extracted entities
- `Keyword`: Extracted keywords

**Relationships:**
- `(Document)-[:HAS_CHUNK]->(Chunk)`
- `(Chunk)-[:MENTIONS]->(Entity)`
- `(Chunk)-[:HAS_KEYWORD]->(Keyword)`

### Useful Cypher Queries

```cypher
// View database schema
CALL db.schema.visualization()

// Count nodes by type
MATCH (n) RETURN labels(n), count(n)

// Find most mentioned entities
MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
RETURN e.value, e.type, count(c) as mentions
ORDER BY mentions DESC
LIMIT 10

// Find documents sharing entities
MATCH (d1:Document)-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(e:Entity)
      <-[:MENTIONS]-(:Chunk)<-[:HAS_CHUNK]-(d2:Document)
WHERE d1 <> d2
RETURN d1.filename, d2.filename, collect(DISTINCT e.value) as shared_entities

// Search by keyword
MATCH (c:Chunk)-[:HAS_KEYWORD]->(k:Keyword {value: "machine"})
RETURN c.content
LIMIT 5
```

### Vector Similarity in Neo4j 5.11+

If using Neo4j 5.11 or later with vector index support:

```cypher
// Vector similarity search
CALL db.index.vector.queryNodes('chunk_embedding_index', 5, $queryVector)
YIELD node, score
RETURN node.content, score
```

## Advanced Usage

### Custom Embedding Models

```python
config = Config(
    embedding_model="sentence-transformers/all-mpnet-base-v2",
    embedding_dimension=768,
    device="cuda"  # Use GPU if available
)
```

### Batch Processing Large Datasets

```python
config = Config(
    batch_size=64,  # Process 64 chunks at a time
    num_workers=8   # Use 8 parallel workers
)
```

### Multiple Databases

```python
# Create separate collections
config1 = Config(collection_name="legal_docs", vector_db_type="neo4j")
config2 = Config(collection_name="medical_docs", vector_db_type="neo4j")

pipeline1 = GraphRAGPipeline(config1)
pipeline2 = GraphRAGPipeline(config2)
```

### Custom Chunking Strategies

```python
# Sentence-based chunking
config = Config(
    chunking_strategy="sentence",
    chunk_size=1024,
    chunk_overlap=100
)

# Paragraph-based chunking
config = Config(
    chunking_strategy="paragraph",
    chunk_size=2048
)

# Token-based chunking (requires LangChain)
config = Config(
    chunking_strategy="token",
    chunk_size=512
)
```

### Monitoring Pipeline Progress

```python
import logging

# Enable detailed logging
logging.basicConfig(level=logging.INFO)

config = Config(
    enable_logging=True,
    log_level="INFO"
)

pipeline = GraphRAGPipeline(config)
stats = pipeline.run()

# Access stage statistics
print(stats['stages']['preprocessing'])
print(stats['stages']['chunking'])
print(stats['stages']['embedding'])
```

### Database Statistics

```python
# Get current database stats
stats = pipeline.get_database_stats()

print(f"Database Type: {stats['db_type']}")
print(f"Total Chunks: {stats.get('total_chunks', 0)}")
print(f"Total Documents: {stats.get('total_documents', 0)}")
print(f"Total Entities: {stats.get('total_entities', 0)}")
```

## Troubleshooting

### Common Issues

**Issue: "langchain not available"**
- Solution: Install LangChain: `pip install langchain langchain-community`
- Or use fallback methods (less efficient but works)

**Issue: "neo4j driver not available"**
- Solution: Install Neo4j driver: `pip install neo4j`
- Or use alternative: `vector_db_type="memory"`

**Issue: "sentence-transformers not installed"**
- Solution: Install: `pip install sentence-transformers`
- Or use fallback embeddings (less accurate)

**Issue: "No documents loaded"**
- Check file paths are correct
- Verify file formats are supported
- Check file permissions

**Issue: Neo4j connection failed**
- Verify Neo4j is running: `docker ps` or check Neo4j Desktop
- Check environment variables are set correctly
- Verify credentials are correct

### Performance Tips

1. **Use GPU for embeddings**: Set `device="cuda"` if available
2. **Increase batch size**: For faster processing: `batch_size=128`
3. **Optimize chunk size**: Balance between context and granularity
4. **Use Neo4j indexes**: For faster queries on large datasets
5. **Enable logging**: Monitor pipeline performance

## Next Steps

- Explore the [example.py](example.py) for more use cases
- Read the full API documentation in code docstrings
- Check out the [tests](tests/) for usage patterns
- Experiment with different configurations
- Integrate with your own applications

For more help, open an issue on GitHub!
