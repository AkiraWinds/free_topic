# GraphRAG Database Setup Project

A comprehensive system for building knowledge graphs with Retrieval-Augmented Generation (RAG) capabilities. This project provides a complete pipeline for data preprocessing, cleaning, chunking, embedding, metadata management, and graph database storage using **Neo4j** and **LangChain**.

## Features

### 🔄 Complete Data Pipeline
- **Data Preprocessing**: Load data from multiple sources (local files, Hugging Face datasets)
- **Data Cleaning**: Text normalization, HTML removal, duplicate detection
- **Intelligent Chunking**: LangChain-powered text splitting with multiple strategies
- **Vector Embeddings**: Generate semantic embeddings using sentence-transformers
- **Metadata Management**: Extract entities, keywords, and enrich metadata
- **Graph Database**: Store and query using Neo4j with relationship mapping

### 🛠️ Technologies Used
- **LangChain**: Document loaders and text splitters
- **Neo4j**: Graph database for storing nodes and relationships
- **Sentence Transformers**: Semantic embeddings generation
- **Pandas**: Data manipulation and HuggingFace dataset loading
- **NumPy**: Numerical operations for embeddings

### 📊 Supported Data Sources
- Local files (TXT, JSON, CSV, PDF)
- Directories with multiple files
- Hugging Face datasets (e.g., Australian Legal Corpus)
- Custom data loaders

## Installation

### Prerequisites
- Python 3.8+
- Neo4j Database (optional, but recommended)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Setup Neo4j (Recommended)

1. **Install Neo4j**:
   - Download from [neo4j.com/download](https://neo4j.com/download/)
   - Or use Docker:
     ```bash
     docker run -d \
       --name neo4j \
       -p 7474:7474 -p 7687:7687 \
       -e NEO4J_AUTH=neo4j/password \
       neo4j:latest
     ```

2. **Set Environment Variables**:
   ```bash
   export NEO4J_URI=bolt://localhost:7687
   export NEO4J_USERNAME=neo4j
   export NEO4J_PASSWORD=password
   ```

3. **Access Neo4j Browser**: http://localhost:7474

## Quick Start

### Basic Usage

```python
from graphrag_db import Config, GraphRAGPipeline

# Create configuration
config = Config(
    input_data_path="data/sample",
    chunk_size=512,
    chunking_strategy="fixed",
    vector_db_type="neo4j"  # or "memory", "chromadb", "faiss"
)

# Initialize and run pipeline
pipeline = GraphRAGPipeline(config)
stats = pipeline.run()

# Query the database
results = pipeline.query("your search query", top_k=5)

# Explore the knowledge graph
graph = pipeline.query_graph("EntityName", depth=2)

# Cleanup
pipeline.close()
```

### Load Australian Legal Corpus

```python
from graphrag_db import Config, GraphRAGPipeline

# Configure for legal documents
config = Config(
    chunk_size=1024,
    chunking_strategy="paragraph",
    vector_db_type="neo4j",
    extract_entities=True,
    extract_keywords=True
)

# Initialize pipeline
pipeline = GraphRAGPipeline(config)

# Load and process the corpus
# Note: Requires Hugging Face authentication (huggingface-cli login)
stats = pipeline.load_australian_legal_corpus()

# Query legal documents
results = pipeline.query("contract dispute resolution", top_k=5)
```

## Configuration Options

### Data Preprocessing
- `input_data_path`: Path to input data
- `supported_formats`: List of file formats (default: ["txt", "json", "csv", "pdf"])
- `encoding`: Text encoding (default: "utf-8")

### Data Cleaning
- `remove_html`: Remove HTML tags and entities
- `remove_urls`: Remove URLs from text
- `normalize_whitespace`: Normalize whitespace
- `remove_duplicates`: Remove duplicate documents
- `min_text_length`: Minimum text length after cleaning

### Chunking (LangChain)
- `chunk_size`: Maximum chunk size in characters (default: 512)
- `chunk_overlap`: Overlap between chunks (default: 50)
- `chunking_strategy`: Strategy to use:
  - `"fixed"`: RecursiveCharacterTextSplitter (default)
  - `"sentence"`: Split by sentences
  - `"paragraph"`: Split by paragraphs
  - `"token"`: Token-based splitting
- `preserve_sentences`: Try to preserve sentence boundaries

### Embedding
- `embedding_model`: Model name (default: "sentence-transformers/all-MiniLM-L6-v2")
- `embedding_dimension`: Embedding vector dimension (default: 384)
- `batch_size`: Batch size for embedding generation (default: 32)
- `device`: Device to use ("cpu", "cuda", "mps")
- `normalize_embeddings`: Normalize embedding vectors

### Metadata Management
- `extract_entities`: Extract named entities
- `extract_keywords`: Extract keywords
- `extract_summary`: Generate summaries
- `max_keywords`: Maximum keywords to extract

### Database (Neo4j)
- `vector_db_type`: Database type ("neo4j", "chromadb", "faiss", "memory")
- `db_path`: Path for database storage
- `collection_name`: Collection/graph name
- `distance_metric`: Distance metric ("cosine", "euclidean", "dot")

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GraphRAG Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Data Preprocessing (LangChain Loaders)                │
│     ├── TextLoader, JSONLoader, CSVLoader                  │
│     ├── DirectoryLoader                                     │
│     └── Custom HuggingFace Loader                          │
│                                                             │
│  2. Data Cleaning                                          │
│     ├── HTML/URL removal                                   │
│     ├── Text normalization                                 │
│     └── Duplicate detection                                │
│                                                             │
│  3. Text Chunking (LangChain Splitters)                   │
│     ├── RecursiveCharacterTextSplitter                     │
│     ├── CharacterTextSplitter                              │
│     └── TokenTextSplitter                                  │
│                                                             │
│  4. Embedding Generation                                   │
│     ├── Sentence Transformers                              │
│     └── Batch processing                                   │
│                                                             │
│  5. Metadata Enrichment                                    │
│     ├── Entity extraction                                  │
│     ├── Keyword extraction                                 │
│     └── Statistics computation                             │
│                                                             │
│  6. Neo4j Graph Storage                                    │
│     ├── Chunk nodes with embeddings                        │
│     ├── Document nodes                                     │
│     ├── Entity nodes                                       │
│     ├── Keyword nodes                                      │
│     └── Relationships (HAS_CHUNK, MENTIONS, HAS_KEYWORD)  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Neo4j Graph Structure

### Nodes
- **Chunk**: Text chunks with embeddings and content
- **Document**: Source documents
- **Entity**: Extracted entities (persons, organizations, etc.)
- **Keyword**: Extracted keywords

### Relationships
- **HAS_CHUNK**: Document → Chunk
- **MENTIONS**: Chunk → Entity
- **HAS_KEYWORD**: Chunk → Keyword

### Example Cypher Queries

```cypher
// Find all chunks from a specific document
MATCH (d:Document {doc_id: "doc_123"})-[:HAS_CHUNK]->(c:Chunk)
RETURN c.content

// Find entities mentioned in chunks
MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
RETURN e.value, e.type, count(c) as mentions
ORDER BY mentions DESC

// Find related documents through shared entities
MATCH (d1:Document)-[:HAS_CHUNK]->(c1:Chunk)-[:MENTIONS]->(e:Entity)
      <-[:MENTIONS]-(c2:Chunk)<-[:HAS_CHUNK]-(d2:Document)
WHERE d1 <> d2
RETURN d1.filename, d2.filename, e.value
```

## Examples

Run the example script to see various use cases:

```bash
python example.py
```

This demonstrates:
1. Basic usage with local files
2. Loading Australian Legal Corpus
3. Neo4j setup and configuration
4. Custom configuration examples

## Project Structure

```
Free_topic/
├── graphrag_db/              # Main package
│   ├── __init__.py
│   ├── config.py             # Configuration management
│   ├── data_preprocessing.py # Data loading (LangChain)
│   ├── data_cleaning.py      # Text cleaning
│   ├── chunking.py           # Text chunking (LangChain)
│   ├── embedding.py          # Embedding generation
│   ├── metadata_manager.py   # Metadata extraction
│   ├── database_manager.py   # Neo4j database operations
│   └── pipeline.py           # Pipeline orchestration
├── data/
│   └── sample/               # Sample data files
├── example.py                # Usage examples
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Advanced Usage

### Custom Configuration

```python
from graphrag_db import Config

config = Config(
    chunk_size=1024,
    chunk_overlap=100,
    chunking_strategy="paragraph",
    embedding_model="sentence-transformers/all-mpnet-base-v2",
    extract_entities=True,
    extract_keywords=True,
    vector_db_type="neo4j"
)

# Save configuration
config.save("config/my_config.json")

# Load configuration
loaded_config = Config.load("config/my_config.json")
```

### Graph Queries

```python
# Query by entity
graph = pipeline.query_graph("Court", relationship_type="MENTIONS", depth=2)
print(f"Found {len(graph['nodes'])} nodes and {len(graph['edges'])} edges")

# Semantic search with filters
results = pipeline.query(
    "contract law",
    top_k=10,
    filters={'doc_id': 'legal_doc_123'}
)
```

## Testing

```bash
pytest tests/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

See LICENSE file for details.

## Acknowledgments

- **LangChain**: Document processing framework
- **Neo4j**: Graph database platform
- **Sentence Transformers**: Embedding models
- **Australian Legal Corpus**: Dataset from Hugging Face (isaacus/open-australian-legal-corpus)

## Support

For issues and questions, please open an issue on GitHub.
