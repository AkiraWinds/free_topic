"""
Database manager module for Neo4j graph database and vector storage operations.
"""

import os
import json
import pickle
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logger.warning("neo4j driver not available. Install with: pip install neo4j")


class DatabaseManager:
    """Handles Neo4j graph database operations including storage and retrieval."""
    
    def __init__(self, config):
        """
        Initialize the database manager.
        
        Args:
            config: Configuration object containing database settings
        """
        self.config = config
        self.db_type = config.vector_db_type
        self.db_path = config.db_path
        self.collection_name = config.collection_name
        self.distance_metric = config.distance_metric
        self.db = None
        self.driver = None
        self._initialize_database()
        
    def _initialize_database(self) -> None:
        """Initialize the database."""
        os.makedirs(self.db_path, exist_ok=True)
        
        if self.db_type == "neo4j":
            self._initialize_neo4j()
        elif self.db_type == "chromadb":
            self._initialize_chromadb()
        elif self.db_type == "faiss":
            self._initialize_faiss()
        elif self.db_type == "qdrant":
            self._initialize_qdrant()
        else:
            logger.warning(f"Unknown database type: {self.db_type}. Using in-memory storage.")
            self._initialize_memory_db()
    
    def _initialize_neo4j(self) -> None:
        """Initialize Neo4j graph database."""
        if not NEO4J_AVAILABLE:
            logger.warning("neo4j driver not available. Using in-memory storage.")
            self._initialize_memory_db()
            return
        
        try:
            # Default Neo4j connection settings
            uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            username = os.getenv("NEO4J_USERNAME", "neo4j")
            password = os.getenv("NEO4J_PASSWORD", "password")
            
            logger.info(f"Connecting to Neo4j at {uri}")
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            
            # Verify connection
            self.driver.verify_connectivity()
            
            # Create indexes and constraints
            self._create_neo4j_schema()
            
            self.db_type = "neo4j"
            logger.info("Neo4j connection established successfully")
            
        except Exception as e:
            logger.error(f"Error connecting to Neo4j: {e}")
            logger.warning("Falling back to in-memory storage")
            self._initialize_memory_db()
    
    def _create_neo4j_schema(self) -> None:
        """Create Neo4j schema (indexes, constraints)."""
        try:
            with self.driver.session() as session:
                # Create constraint for unique chunk IDs
                session.run("""
                    CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
                    FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE
                """)
                
                # Create index for document IDs
                session.run("""
                    CREATE INDEX doc_id_index IF NOT EXISTS
                    FOR (c:Chunk) ON (c.doc_id)
                """)
                
                # Create vector index if supported (Neo4j 5.11+)
                try:
                    session.run(f"""
                        CREATE VECTOR INDEX chunk_embedding_index IF NOT EXISTS
                        FOR (c:Chunk) ON (c.embedding)
                        OPTIONS {{
                            indexConfig: {{
                                `vector.dimensions`: {self.config.embedding_dimension},
                                `vector.similarity_function`: 'cosine'
                            }}
                        }}
                    """)
                    logger.info("Created vector index for embeddings")
                except Exception as e:
                    logger.warning(f"Could not create vector index (requires Neo4j 5.11+): {e}")
                
                logger.info("Neo4j schema created successfully")
        except Exception as e:
            logger.error(f"Error creating Neo4j schema: {e}")
    
    def _initialize_chromadb(self) -> None:
        """Initialize ChromaDB."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            logger.info(f"Initializing ChromaDB at {self.db_path}")
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.db = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_metric}
            )
            logger.info(f"ChromaDB collection '{self.collection_name}' ready")
            
        except ImportError:
            logger.warning("chromadb not installed. Using in-memory storage.")
            self._initialize_memory_db()
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}. Using in-memory storage.")
            self._initialize_memory_db()
    
    def _initialize_faiss(self) -> None:
        """Initialize FAISS index."""
        try:
            import faiss
            
            logger.info("Initializing FAISS index")
            # Will be created when first vectors are added
            self.db = {
                'index': None,
                'metadata': [],
                'ids': []
            }
            logger.info("FAISS index ready")
            
        except ImportError:
            logger.warning("faiss not installed. Using in-memory storage.")
            self._initialize_memory_db()
    
    def _initialize_qdrant(self) -> None:
        """Initialize Qdrant."""
        logger.warning("Qdrant initialization not implemented. Using in-memory storage.")
        self._initialize_memory_db()
    
    def _initialize_memory_db(self) -> None:
        """Initialize simple in-memory database."""
        logger.info("Initializing in-memory database")
        self.db = {
            'vectors': [],
            'metadata': [],
            'ids': [],
            'contents': []
        }
        self.db_type = "memory"
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add chunks with embeddings to the database.
        
        Args:
            chunks: List of chunk dictionaries with 'embedding' field
        """
        if not chunks:
            logger.warning("No chunks to add to database")
            return
        
        if self.db_type == "neo4j":
            self._add_to_neo4j(chunks)
        elif self.db_type == "chromadb":
            self._add_to_chromadb(chunks)
        elif self.db_type == "faiss":
            self._add_to_faiss(chunks)
        elif self.db_type == "memory":
            self._add_to_memory(chunks)
        
        logger.info(f"Added {len(chunks)} chunks to {self.db_type} database")
    
    def _add_to_neo4j(self, chunks: List[Dict[str, Any]]) -> None:
        """Add chunks to Neo4j graph database."""
        if self.driver is None:
            logger.error("Neo4j driver not initialized")
            return
        
        try:
            with self.driver.session() as session:
                for chunk in chunks:
                    # Prepare chunk data
                    chunk_id = chunk['metadata'].get('chunk_id', 'unknown')
                    content = chunk['content']
                    metadata = chunk['metadata']
                    
                    # Convert embedding to list if it's numpy array
                    embedding = chunk.get('embedding', [])
                    if isinstance(embedding, np.ndarray):
                        embedding = embedding.tolist()
                    
                    # Create Chunk node
                    session.run("""
                        MERGE (c:Chunk {chunk_id: $chunk_id})
                        SET c.content = $content,
                            c.doc_id = $doc_id,
                            c.chunk_index = $chunk_index,
                            c.total_chunks = $total_chunks,
                            c.chunk_size = $chunk_size,
                            c.embedding = $embedding,
                            c.source = $source,
                            c.filename = $filename
                    """, 
                        chunk_id=chunk_id,
                        content=content,
                        doc_id=metadata.get('doc_id', ''),
                        chunk_index=metadata.get('chunk_index', 0),
                        total_chunks=metadata.get('total_chunks', 1),
                        chunk_size=metadata.get('chunk_size', len(content)),
                        embedding=embedding,
                        source=metadata.get('source', ''),
                        filename=metadata.get('filename', '')
                    )
                    
                    # Create Document node and relationship
                    doc_id = metadata.get('doc_id', 'unknown')
                    session.run("""
                        MERGE (d:Document {doc_id: $doc_id})
                        SET d.source = $source,
                            d.filename = $filename
                        WITH d
                        MATCH (c:Chunk {chunk_id: $chunk_id})
                        MERGE (d)-[:HAS_CHUNK]->(c)
                    """,
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        source=metadata.get('source', ''),
                        filename=metadata.get('filename', '')
                    )
                    
                    # Extract and create entity nodes if available
                    entities = metadata.get('entities', {})
                    if entities:
                        self._add_entities_to_neo4j(session, chunk_id, entities)
                    
                    # Create keyword nodes if available
                    keywords = metadata.get('keywords', [])
                    if keywords:
                        self._add_keywords_to_neo4j(session, chunk_id, keywords)
            
            logger.info(f"Successfully added {len(chunks)} chunks to Neo4j")
            
        except Exception as e:
            logger.error(f"Error adding chunks to Neo4j: {e}")
    
    def _add_entities_to_neo4j(self, session, chunk_id: str, entities: Dict[str, List[str]]) -> None:
        """Add entity nodes and relationships to Neo4j."""
        for entity_type, entity_list in entities.items():
            for entity_value in entity_list:
                if entity_value:
                    session.run("""
                        MERGE (e:Entity {value: $entity_value, type: $entity_type})
                        WITH e
                        MATCH (c:Chunk {chunk_id: $chunk_id})
                        MERGE (c)-[:MENTIONS]->(e)
                    """,
                        entity_value=entity_value,
                        entity_type=entity_type,
                        chunk_id=chunk_id
                    )
    
    def _add_keywords_to_neo4j(self, session, chunk_id: str, keywords: List[str]) -> None:
        """Add keyword nodes and relationships to Neo4j."""
        for keyword in keywords:
            if keyword:
                session.run("""
                    MERGE (k:Keyword {value: $keyword})
                    WITH k
                    MATCH (c:Chunk {chunk_id: $chunk_id})
                    MERGE (c)-[:HAS_KEYWORD]->(k)
                """,
                    keyword=keyword,
                    chunk_id=chunk_id
                )
    
    def _add_to_chromadb(self, chunks: List[Dict[str, Any]]) -> None:
        """Add chunks to ChromaDB."""
        if self.db is None:
            logger.error("ChromaDB not initialized")
            return
        
        ids = []
        embeddings = []
        metadatas = []
        documents = []
        
        for chunk in chunks:
            chunk_id = chunk['metadata'].get('chunk_id', str(len(ids)))
            ids.append(chunk_id)
            
            # Convert numpy array to list if needed
            embedding = chunk.get('embedding', [])
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            embeddings.append(embedding)
            
            # Prepare metadata (ChromaDB requires string values)
            metadata = {}
            for key, value in chunk['metadata'].items():
                if isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
                elif isinstance(value, list) and all(isinstance(x, str) for x in value):
                    metadata[key] = ", ".join(value)
                else:
                    metadata[key] = str(value)
            metadatas.append(metadata)
            
            documents.append(chunk['content'])
        
        try:
            self.db.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
        except Exception as e:
            logger.error(f"Error adding to ChromaDB: {e}")
    
    def _add_to_faiss(self, chunks: List[Dict[str, Any]]) -> None:
        """Add chunks to FAISS index."""
        import faiss
        
        embeddings = []
        for chunk in chunks:
            embedding = chunk.get('embedding', [])
            if isinstance(embedding, np.ndarray):
                embeddings.append(embedding)
            else:
                embeddings.append(np.array(embedding))
        
        if not embeddings:
            return
        
        embeddings_array = np.vstack(embeddings).astype('float32')
        
        # Create index if it doesn't exist
        if self.db['index'] is None:
            dimension = embeddings_array.shape[1]
            if self.distance_metric == "cosine":
                self.db['index'] = faiss.IndexFlatIP(dimension)  # Inner product for cosine
            else:
                self.db['index'] = faiss.IndexFlatL2(dimension)  # L2 distance
        
        # Normalize for cosine similarity
        if self.distance_metric == "cosine":
            faiss.normalize_L2(embeddings_array)
        
        # Add vectors
        self.db['index'].add(embeddings_array)
        
        # Store metadata
        for chunk in chunks:
            self.db['metadata'].append(chunk['metadata'])
            self.db['ids'].append(chunk['metadata'].get('chunk_id', f"chunk_{len(self.db['ids'])}"))
    
    def _add_to_memory(self, chunks: List[Dict[str, Any]]) -> None:
        """Add chunks to in-memory storage."""
        for chunk in chunks:
            embedding = chunk.get('embedding', [])
            if isinstance(embedding, np.ndarray):
                self.db['vectors'].append(embedding)
            else:
                self.db['vectors'].append(np.array(embedding))
            
            self.db['metadata'].append(chunk['metadata'])
            self.db['ids'].append(chunk['metadata'].get('chunk_id', f"chunk_{len(self.db['ids'])}"))
            self.db['contents'].append(chunk['content'])
    
    def query(self, query_embedding: np.ndarray, top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Query the database for similar vectors.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filters: Optional filters (e.g., {'doc_id': 'doc_123'})
            
        Returns:
            List of result dictionaries with content, metadata, and similarity score
        """
        if self.db_type == "neo4j":
            return self._query_neo4j(query_embedding, top_k, filters)
        elif self.db_type == "chromadb":
            return self._query_chromadb(query_embedding, top_k)
        elif self.db_type == "faiss":
            return self._query_faiss(query_embedding, top_k)
        elif self.db_type == "memory":
            return self._query_memory(query_embedding, top_k)
        
        return []
    
    def _query_neo4j(self, query_embedding: np.ndarray, top_k: int, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Query Neo4j for similar chunks using vector similarity or manual calculation."""
        if self.driver is None:
            return []
        
        try:
            with self.driver.session() as session:
                # Try vector index search first (Neo4j 5.11+)
                try:
                    query_vector = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
                    
                    # Build filter clause
                    filter_clause = ""
                    if filters:
                        conditions = [f"c.{key} = ${key}" for key in filters.keys()]
                        filter_clause = "WHERE " + " AND ".join(conditions)
                    
                    result = session.run(f"""
                        MATCH (c:Chunk)
                        {filter_clause}
                        WITH c, vector.similarity.cosine(c.embedding, $query_vector) AS similarity
                        WHERE c.embedding IS NOT NULL
                        RETURN c.chunk_id AS id, 
                               c.content AS content,
                               c.doc_id AS doc_id,
                               c.source AS source,
                               c.chunk_index AS chunk_index,
                               similarity
                        ORDER BY similarity DESC
                        LIMIT $top_k
                    """, query_vector=query_vector, top_k=top_k, **(filters or {}))
                    
                    results = []
                    for record in result:
                        results.append({
                            'id': record['id'],
                            'content': record['content'],
                            'metadata': {
                                'doc_id': record.get('doc_id', ''),
                                'source': record.get('source', ''),
                                'chunk_index': record.get('chunk_index', 0)
                            },
                            'similarity': record['similarity']
                        })
                    
                    if results:
                        return results
                        
                except Exception as e:
                    logger.warning(f"Vector similarity search not available: {e}")
                
                # Fallback: retrieve all chunks and calculate similarity manually
                logger.info("Using manual similarity calculation")
                filter_clause = ""
                if filters:
                    conditions = [f"c.{key} = ${key}" for key in filters.keys()]
                    filter_clause = "WHERE " + " AND ".join(conditions)
                
                result = session.run(f"""
                    MATCH (c:Chunk)
                    {filter_clause}
                    WHERE c.embedding IS NOT NULL
                    RETURN c.chunk_id AS id,
                           c.content AS content,
                           c.doc_id AS doc_id,
                           c.source AS source,
                           c.chunk_index AS chunk_index,
                           c.embedding AS embedding
                """, **(filters or {}))
                
                # Calculate similarities manually
                chunks = []
                for record in result:
                    embedding = np.array(record['embedding'])
                    similarity = self._cosine_similarity(query_embedding, embedding)
                    chunks.append({
                        'id': record['id'],
                        'content': record['content'],
                        'metadata': {
                            'doc_id': record.get('doc_id', ''),
                            'source': record.get('source', ''),
                            'chunk_index': record.get('chunk_index', 0)
                        },
                        'similarity': float(similarity)
                    })
                
                # Sort by similarity and return top k
                chunks.sort(key=lambda x: x['similarity'], reverse=True)
                return chunks[:top_k]
                
        except Exception as e:
            logger.error(f"Error querying Neo4j: {e}")
            return []
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def query_graph(self, entity: str, relationship_type: Optional[str] = None, depth: int = 2) -> Dict[str, Any]:
        """
        Query the knowledge graph for entity relationships.
        
        Args:
            entity: Entity value to search for
            relationship_type: Optional relationship type to filter
            depth: Depth of graph traversal
            
        Returns:
            Dictionary with graph structure (nodes and edges)
        """
        if self.db_type != "neo4j" or self.driver is None:
            logger.warning("Graph queries only supported with Neo4j")
            return {'nodes': [], 'edges': []}
        
        try:
            with self.driver.session() as session:
                rel_filter = f":{relationship_type}" if relationship_type else ""
                
                result = session.run(f"""
                    MATCH path = (e:Entity {{value: $entity}})-[r{rel_filter}*1..{depth}]-(connected)
                    RETURN nodes(path) AS nodes, relationships(path) AS relationships
                    LIMIT 100
                """, entity=entity)
                
                nodes = []
                edges = []
                node_ids = set()
                
                for record in result:
                    # Process nodes
                    for node in record['nodes']:
                        node_id = node.element_id
                        if node_id not in node_ids:
                            node_ids.add(node_id)
                            nodes.append({
                                'id': node_id,
                                'labels': list(node.labels),
                                'properties': dict(node)
                            })
                    
                    # Process relationships
                    for rel in record['relationships']:
                        edges.append({
                            'type': rel.type,
                            'start': rel.start_node.element_id,
                            'end': rel.end_node.element_id,
                            'properties': dict(rel)
                        })
                
                return {'nodes': nodes, 'edges': edges}
                
        except Exception as e:
            logger.error(f"Error querying graph: {e}")
            return {'nodes': [], 'edges': []}
    
    def _query_chromadb(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Query ChromaDB."""
        if self.db is None:
            return []
        
        try:
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
            
            results = self.db.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            # Format results
            formatted_results = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        'id': results['ids'][0][i],
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if 'distances' in results else None
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")
            return []
    
    def _query_faiss(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Query FAISS index."""
        if self.db['index'] is None or self.db['index'].ntotal == 0:
            return []
        
        import faiss
        
        # Prepare query vector
        query_vector = query_embedding.reshape(1, -1).astype('float32')
        
        # Normalize for cosine similarity
        if self.distance_metric == "cosine":
            faiss.normalize_L2(query_vector)
        
        # Search
        distances, indices = self.db['index'].search(query_vector, min(top_k, self.db['index'].ntotal))
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0:  # Valid index
                results.append({
                    'id': self.db['ids'][idx],
                    'metadata': self.db['metadata'][idx],
                    'distance': float(distances[0][i])
                })
        
        return results
    
    def _query_memory(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Query in-memory database."""
        if not self.db['vectors']:
            return []
        
        # Calculate similarities
        vectors = np.vstack(self.db['vectors'])
        
        if self.distance_metric == "cosine":
            # Cosine similarity
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            similarities = np.dot(vectors_norm, query_norm)
            top_indices = np.argsort(similarities)[::-1][:top_k]
        else:
            # Euclidean distance
            distances = np.linalg.norm(vectors - query_embedding, axis=1)
            top_indices = np.argsort(distances)[:top_k]
            similarities = 1 / (1 + distances)  # Convert distance to similarity
        
        # Format results
        results = []
        for idx in top_indices:
            results.append({
                'id': self.db['ids'][idx],
                'content': self.db['contents'][idx],
                'metadata': self.db['metadata'][idx],
                'similarity': float(similarities[idx]) if self.distance_metric == "cosine" else float(similarities[idx])
            })
        
        return results
    
    def save(self) -> None:
        """Save the database to disk."""
        if self.db_type == "neo4j":
            # Neo4j persists automatically
            logger.info("Neo4j persists automatically, no explicit save needed")
        elif self.db_type == "chromadb":
            # ChromaDB auto-persists
            logger.info("ChromaDB auto-persists, no explicit save needed")
        elif self.db_type in ["faiss", "memory"]:
            save_path = os.path.join(self.db_path, f"{self.collection_name}.pkl")
            with open(save_path, 'wb') as f:
                pickle.dump(self.db, f)
            logger.info(f"Database saved to {save_path}")
    
    def load(self) -> None:
        """Load the database from disk."""
        if self.db_type == "neo4j":
            # Neo4j loads automatically on connection
            logger.info("Neo4j loads automatically on connection")
        elif self.db_type == "chromadb":
            # ChromaDB loads automatically
            logger.info("ChromaDB loads automatically")
        elif self.db_type in ["faiss", "memory"]:
            save_path = os.path.join(self.db_path, f"{self.collection_name}.pkl")
            if os.path.exists(save_path):
                with open(save_path, 'rb') as f:
                    self.db = pickle.load(f)
                logger.info(f"Database loaded from {save_path}")
    
    def close(self) -> None:
        """Close database connections."""
        if self.db_type == "neo4j" and self.driver is not None:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {
            'db_type': self.db_type,
            'collection_name': self.collection_name,
            'distance_metric': self.distance_metric
        }
        
        if self.db_type == "neo4j" and self.driver is not None:
            try:
                with self.driver.session() as session:
                    result = session.run("""
                        MATCH (c:Chunk)
                        RETURN count(c) AS chunk_count
                    """)
                    record = result.single()
                    stats['total_chunks'] = record['chunk_count'] if record else 0
                    
                    result = session.run("""
                        MATCH (d:Document)
                        RETURN count(d) AS doc_count
                    """)
                    record = result.single()
                    stats['total_documents'] = record['doc_count'] if record else 0
                    
                    result = session.run("""
                        MATCH (e:Entity)
                        RETURN count(e) AS entity_count
                    """)
                    record = result.single()
                    stats['total_entities'] = record['entity_count'] if record else 0
                    
                    result = session.run("""
                        MATCH (k:Keyword)
                        RETURN count(k) AS keyword_count
                    """)
                    record = result.single()
                    stats['total_keywords'] = record['keyword_count'] if record else 0
            except Exception as e:
                logger.error(f"Error getting Neo4j stats: {e}")
        elif self.db_type == "chromadb" and self.db is not None:
            stats['total_vectors'] = self.db.count()
        elif self.db_type == "faiss" and self.db['index'] is not None:
            stats['total_vectors'] = self.db['index'].ntotal
        elif self.db_type == "memory":
            stats['total_vectors'] = len(self.db['vectors'])
        
        return stats
