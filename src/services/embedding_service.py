"""Embedding service with caching and batch processing."""

import asyncio
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from sentence_transformers import SentenceTransformer
import hashlib
import pickle

from src.core.config import settings
from src.core.cache import cache
from src.core.monitoring import monitoring_service
from src.core.models import ChunkModel
from src.core.database import get_db_session
from sqlalchemy import select
from .vector_db_client import ResilientVectorStore

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating and managing embeddings."""
    
    def __init__(self):
        # Passage and query models (may be the same)
        self.passage_model: Optional[SentenceTransformer] = None
        self.query_model: Optional[SentenceTransformer] = None
        self.passage_model_name = settings.embedding_model_passage or settings.embedding_model
        self.query_model_name = settings.embedding_model_query or settings.embedding_model
        self.dimension = settings.embedding_dimension
        
    async def initialize(self):
        """Initialize the embedding model."""
        try:
            # Load model in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            self.passage_model, self.query_model = await loop.run_in_executor(
                None,
                self._load_models
            )
            logger.info(f"Embedding models loaded: passage={self.passage_model_name}, query={self.query_model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
            
    def _load_models(self) -> Tuple[SentenceTransformer, SentenceTransformer]:
        """Load passage and query sentence transformer models."""
        passage = SentenceTransformer(self.passage_model_name)
        # If query model name equals passage, reuse the same instance
        if self.query_model_name == self.passage_model_name:
            query = passage
        else:
            query = SentenceTransformer(self.query_model_name)
        return passage, query
        
    async def embed_text(self, text: str, *, is_query: bool = False) -> np.ndarray:
        """Generate embedding for a single text with optional query instruction."""
        if not self.passage_model or not self.query_model:
            await self.initialize()
        
        # Apply instruction formatting for models that require it
        formatted_text = self._format_with_instruction(text, is_query=is_query)

        # Check cache first
        cache_key = self._get_cache_key(formatted_text, is_query=is_query)
        cached_embedding = await cache.get(cache_key)
        
        if cached_embedding is not None:
            return cached_embedding
            
        # Generate embedding
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            self._generate_embedding,
            formatted_text,
            is_query
        )
        
        # Cache the result
        await cache.set(cache_key, embedding, ttl=settings.query_embedding_cache_ttl)
        
        return embedding
        
    def _generate_embedding(self, text: str, is_query: bool) -> np.ndarray:
        """Generate embedding using the appropriate model."""
        model = self.query_model if is_query else self.passage_model
        return model.encode(text, convert_to_numpy=True)
        
    async def embed_batch(self, texts: List[str], *, is_query: bool = False) -> List[np.ndarray]:
        """Generate embeddings for a batch of texts (query or passage)."""
        if not self.passage_model or not self.query_model:
            await self.initialize()
        
        # Apply instruction formatting
        formatted_texts = [self._format_with_instruction(t, is_query=is_query) for t in texts]

        # Check cache for each text
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(formatted_texts):
            cache_key = self._get_cache_key(text, is_query=is_query)
            cached_embedding = await cache.get(cache_key)
            
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
            else:
                embeddings.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)
                
        # Generate embeddings for uncached texts
        if uncached_texts:
            loop = asyncio.get_event_loop()
            new_embeddings = await loop.run_in_executor(
                None,
                self._generate_batch_embeddings,
                uncached_texts,
                is_query
            )
            
            # Cache new embeddings and update results
            for idx, embedding in zip(uncached_indices, new_embeddings):
                # L2-normalize embeddings for consistency
                if embedding is not None:
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                embeddings[idx] = embedding
                cache_key = self._get_cache_key(formatted_texts[idx], is_query=is_query)
                await cache.set(cache_key, embedding, ttl=settings.query_embedding_cache_ttl)
                
        return embeddings
        
    def _generate_batch_embeddings(self, texts: List[str], is_query: bool) -> List[np.ndarray]:
        """Generate embeddings for a batch of texts."""
        model = self.query_model if is_query else self.passage_model
        return model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        
    async def embed_chunks(self, document_id: str) -> Dict[str, Any]:
        """Generate embeddings for all chunks of a document."""
        start_time = time.time()

        async with get_db_session() as session:
            # Get all chunks for the document
            result = await session.execute(
                select(ChunkModel)
                .where(ChunkModel.document_id == document_id)
                .order_by(ChunkModel.chunk_index)
            )
            chunks = result.scalars().all()

            if not chunks:
                return {"embedded": 0, "failed": 0}

            # Extract texts
            texts = [chunk.content for chunk in chunks]

            try:
                # Generate embeddings in batches
                batch_size = settings.batch_size
                all_embeddings = []

                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_embeddings = await self.embed_batch(batch_texts)
                    all_embeddings.extend(batch_embeddings)

                # Store embeddings (this would typically go to a vector database)
                embedded_count = 0
                for chunk, embedding in zip(chunks, all_embeddings):
                    if embedding is not None:
                        # Store embedding ID or vector (implementation depends on vector DB)
                        chunk.embedding_id = self._store_embedding(
                            str(chunk.id),
                            embedding
                        )
                        embedded_count += 1

                await session.commit()

                # Track embedding metrics
                embedding_duration = time.time() - start_time
                await monitoring_service.track_embedding_generation(
                    count=embedded_count,
                    duration=embedding_duration
                )

                # Log successful embedding generation
                await monitoring_service.log_event(
                    "embeddings_generated",
                    {
                        "document_id": document_id,
                        "chunks_count": len(chunks),
                        "embedded_count": embedded_count,
                        "duration": embedding_duration
                    }
                )

                return {
                    "embedded": embedded_count,
                    "failed": len(chunks) - embedded_count
                }

            except Exception as e:
                # Track failed embedding generation
                embedding_duration = time.time() - start_time
                await monitoring_service.track_embedding_generation(
                    count=0,
                    duration=embedding_duration
                )

                await monitoring_service.log_event(
                    "embedding_generation_failed",
                    {
                        "document_id": document_id,
                        "chunks_count": len(chunks),
                        "error": str(e)
                    },
                    level="error"
                )

                logger.error(f"Error embedding chunks for document {document_id}: {e}")
                return {"embedded": 0, "failed": len(chunks)}
                
    def _store_embedding(self, chunk_id: str, embedding: np.ndarray) -> str:
        """Store embedding in vector database (placeholder)."""
        # This is a placeholder - in a real implementation, you would:
        # 1. Store in ChromaDB, Pinecone, Weaviate, etc.
        # 2. Return the vector database ID
        
        # For now, just return a hash of the embedding
        embedding_bytes = embedding.tobytes()
        return hashlib.sha256(embedding_bytes).hexdigest()[:16]
        
    async def find_similar_chunks(
        self, 
        query_text: str, 
        k: int = 5,
        similarity_threshold: float = None
    ) -> List[Tuple[str, float]]:
        """Find similar chunks using embedding similarity."""
        if similarity_threshold is None:
            similarity_threshold = settings.similarity_threshold
            
        # Generate query embedding
            query_embedding = await self.embed_text(query_text, is_query=True)
        
        # This is a placeholder for vector similarity search
        # In a real implementation, you would:
        # 1. Query the vector database (ChromaDB, Pinecone, etc.)
        # 2. Return chunk IDs with similarity scores
        
        # For now, return empty list
        return []
        
    def _get_cache_key(self, text: str, *, is_query: bool = False) -> str:
        """Generate cache key for text with versioning and model role."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        role = "query" if is_query else "passage"
        model_name = self.query_model_name if is_query else self.passage_model_name
        version = settings.embedding_version
        return f"embedding:v{version}:{role}:{model_name}:{text_hash}"

    def _format_with_instruction(self, text: str, *, is_query: bool = False) -> str:
        """Apply instruction prefix for BGE/E5-style models when configured."""
        prefix = settings.embedding_query_instruction if is_query else settings.embedding_passage_instruction
        # If prefix is empty or already applied, avoid duplication
        if not prefix:
            return text
        if text.startswith(prefix):
            return text
        return f"{prefix}{text}"
        
    async def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding service statistics."""
        # This would typically query the vector database for stats
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "total_embeddings": 0,  # Placeholder
            "cache_hit_rate": 0.0   # Placeholder
        }
        
    async def clear_embeddings_cache(self) -> int:
        """Clear embeddings cache."""
        pattern = f"embedding:{self.model_name}:*"
        return await cache.clear_pattern(pattern)


class VectorStore:
    """Vector store interface for managing embeddings."""
    
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.collection_name = "document_chunks"
        # Wrap with resilient client for retry and backpressure
        self._resilient = ResilientVectorStore(self)
    
    def _create_chroma_client(self):
        """Create a ChromaDB client with consistent settings."""
        import chromadb
        # Use new ChromaDB client architecture (no Settings)
        return chromadb.PersistentClient(
            path=settings.vector_db_path
        )
        
    async def add_documents(self, chunks: List[ChunkModel]) -> List[str]:
        """Add document chunks to vector store using ChromaDB."""
        return await self._resilient.add_documents(chunks)
    
    async def _add_documents_impl(self, chunks: List[ChunkModel]) -> List[str]:
        """Internal implementation of add_documents."""
        try:
            # Initialize ChromaDB client with telemetry disabled
            chroma_client = self._create_chroma_client()
            
            # Get or create collection
            try:
                collection = chroma_client.get_collection(name=self.collection_name)
            except:
                collection = chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={
                        "hnsw:space": "cosine"
                    }
                )
            
            # Extract texts and metadata
            texts = [chunk.content for chunk in chunks]
            metadatas = []
            ids = []
            
            for chunk in chunks:
                # Flatten metadata for ChromaDB compatibility
                base_metadata = {
                    "chunk_id": str(chunk.id),
                    "document_id": str(chunk.document_id),
                    "chunk_index": chunk.chunk_index,
                }
                
                # Add extra metadata, flattening nested dicts and converting to strings
                if chunk.extra_metadata:
                    for key, value in chunk.extra_metadata.items():
                        if isinstance(value, dict):
                            # Flatten nested dictionaries
                            for nested_key, nested_value in value.items():
                                flat_key = f"{key}_{nested_key}".replace('/', '_').replace(' ', '_')
                                base_metadata[flat_key] = str(nested_value) if nested_value is not None else ""
                        elif isinstance(value, (list, tuple)):
                            # Convert lists/tuples to comma-separated strings
                            base_metadata[key] = ",".join(str(v) for v in value)
                        elif value is not None:
                            # Convert other types to strings
                            base_metadata[key] = str(value)
                
                metadatas.append(base_metadata)
                # Use the raw chunk ID to ensure consistency with the database
                ids.append(str(chunk.id))
                
            # Generate embeddings
            embeddings = await self.embedding_service.embed_batch(texts, is_query=False)
            
            # Filter out chunks with failed embeddings
            valid_data = []
            for i, (chunk, embedding, metadata, chunk_id) in enumerate(zip(chunks, embeddings, metadatas, ids)):
                if embedding is not None:
                    valid_data.append({
                        'id': chunk_id,
                        'embedding': embedding.tolist(),
                        'metadata': metadata,
                        'document': chunk.content
                    })
            
            if valid_data:
                # Add to ChromaDB
                # Use upsert to avoid duplicates on re-embedding
                collection.upsert(
                    ids=[item['id'] for item in valid_data],
                    embeddings=[item['embedding'] for item in valid_data],
                    metadatas=[item['metadata'] for item in valid_data],
                    documents=[item['document'] for item in valid_data]
                )
                
            return [item['id'] for item in valid_data]
            
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return []
        
    async def search(
        self, 
        query_text: str, 
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents using ChromaDB."""
        return await self._resilient.search(query_text, k, filters)
    
    async def _search_impl(
        self, 
        query_text: str, 
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Internal implementation of search."""
        try:
            # Initialize ChromaDB client with telemetry disabled
            chroma_client = self._create_chroma_client()
            
            # Get or create collection
            try:
                collection = chroma_client.get_collection(name=self.collection_name)
            except:
                # Collection doesn't exist, return empty results
                logger.warning(f"Collection {self.collection_name} not found in ChromaDB")
                return []
            
            # Generate query embedding
            query_embedding = await self.embedding_service.embed_text(query_text, is_query=True)
            if query_embedding is None:
                return []
            
            # Search in ChromaDB
            # Increase n_results query-side; thresholding will be applied later
            # Build where filter from provided filters if present
            where_filter = None
            if filters and isinstance(filters, dict):
                # Accept direct 'where' or build from 'chunk_metadata' keys
                if 'where' in filters and isinstance(filters['where'], dict):
                    where_filter = filters['where']
                elif 'chunk_metadata' in filters and isinstance(filters['chunk_metadata'], dict):
                    # Chroma expects flat keys; our metadata is flattened in add_documents
                    where_filter = {k: str(v) for k, v in filters['chunk_metadata'].items()}

            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=max(k * 3, 10),
                include=['metadatas', 'distances', 'documents'],
                where=where_filter
            )
            
            # Convert results to expected format, normalizing ID to DB chunk ID
            search_results = []
            if results['ids'] and results['ids'][0]:
                for i, raw_id in enumerate(results['ids'][0]):
                    distance = results['distances'][0][i] if results['distances'] else 0
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    document_text = results['documents'][0][i] if results.get('documents') else None

                    # Prefer explicit metadata chunk_id if present
                    chunk_id = metadata.get('chunk_id') if isinstance(metadata, dict) else None
                    if not chunk_id:
                        # Normalize legacy IDs like "chunk_<uuid>" to "<uuid>"
                        chunk_id = raw_id
                        if isinstance(chunk_id, str) and chunk_id.startswith('chunk_'):
                            chunk_id = chunk_id[len('chunk_'):]
                    
                    # Convert distance to similarity score (1 - distance)
                    score = max(0, 1 - distance)
                    
                    search_results.append({
                        'chunk_id': chunk_id,
                        'score': score,
                        'metadata': metadata,
                        'document': document_text,
                        'raw_id': raw_id
                    })
            
            return search_results
            
        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            # Fallback to empty results
            return []
        
    async def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a document from vector store."""
        return await self._resilient.delete_document(document_id)
    
    async def _delete_document_impl(self, document_id: str) -> int:
        """Internal implementation of delete_document."""
        try:
            import chromadb
            
            # Initialize ChromaDB client
            chroma_client = self._create_chroma_client()
            
            try:
                collection = chroma_client.get_collection(name=self.collection_name)
                
                # First, try to get the count of items to delete for logging
                try:
                    # Query to get all items with this document_id
                    query_result = collection.get(
                        where={"document_id": document_id},
                        include=["metadatas"]
                    )
                    item_count = len(query_result.get('ids', []))
                    
                    if item_count == 0:
                        logger.info(f"No vectors found for document {document_id}")
                        return 0
                    
                    logger.info(f"Deleting {item_count} vectors for document {document_id}")
                except Exception as e:
                    logger.warning(f"Could not count items for document {document_id}: {e}")
                    item_count = "unknown"
                
                # Delete all chunks for this document using where clause
                # This is more efficient than deleting by individual IDs
                result = collection.delete(
                    where={"document_id": document_id}
                )
                
                deleted_count = len(result.get('ids', []))
                logger.info(f"Successfully deleted {deleted_count} vectors for document {document_id}")
                return deleted_count
                
            except Exception as e:
                logger.warning(f"Collection not found or empty during delete: {e}")
                return 0
                
        except Exception as e:
            logger.error(f"ChromaDB delete failed: {e}")
            return 0
        
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the vector collection."""
        base_info = {
            "collection_name": self.collection_name,
            "total_vectors": 0,  # Placeholder
            "dimension": self.embedding_service.dimension
        }
        
        # Add operation statistics
        try:
            stats = await self._resilient.get_operation_stats()
            base_info["operation_stats"] = stats
        except Exception as e:
            logger.warning(f"Could not get operation stats: {e}")
        
        return base_info

    async def migrate_collection_ids(self) -> Dict[str, Any]:
        """Migrate existing vectors to use consistent DB chunk IDs.

        Strategy:
        - Read all items from the collection (ids, embeddings, documents, metadatas)
        - Determine canonical id per item:
            * Prefer metadata['chunk_id'] if present
            * Else, strip legacy prefix 'chunk_' if present
            * Else, keep current id
        - For items where canonical id differs from current id, upsert a new item with canonical id
          (same embedding/document/metadata, ensure metadata['chunk_id'] is set), then delete old id
        """
        try:
            import chromadb
        except Exception as e:  # pragma: no cover
            logger.error(f"ChromaDB not available for migration: {e}")
            return {"migrated": 0, "errors": 1}

        chroma_client = chromadb.PersistentClient(
            path=settings.vector_db_path
        )
        try:
            collection = chroma_client.get_collection(name=self.collection_name)
        except Exception:
            logger.warning(f"Collection {self.collection_name} not found; nothing to migrate")
            return {"migrated": 0, "errors": 0}

        migrated_count = 0
        error_count = 0
        batch_size = 1000
        offset = 0

        while True:
            try:
                # Attempt paginated get; if offset unsupported, fetch all once
                results = collection.get(
                    where={},
                    include=["embeddings", "metadatas", "documents"],
                    limit=batch_size,
                    offset=offset
                )
            except TypeError:
                # Older clients may not support offset; fetch all and process once
                results = collection.get(where={}, include=["embeddings", "metadatas", "documents"])
                offset = None  # disable pagination

            ids = results.get("ids", []) or []
            if not ids:
                break

            embeddings = results.get("embeddings", []) or []
            documents = results.get("documents", []) or []
            metadatas = results.get("metadatas", []) or []

            to_upsert = []
            to_delete = []

            for current_id, emb, doc, meta in zip(ids, embeddings, documents, metadatas):
                try:
                    meta = meta or {}
                    # Determine canonical id
                    canonical = None
                    if isinstance(meta, dict) and meta.get("chunk_id"):
                        canonical = str(meta["chunk_id"])
                    else:
                        canonical = str(current_id)
                        if isinstance(canonical, str) and canonical.startswith("chunk_"):
                            canonical = canonical[len("chunk_"):]

                    # Ensure metadata has chunk_id
                    meta = dict(meta)
                    meta["chunk_id"] = canonical

                    if str(current_id) == canonical:
                        continue

                    to_upsert.append({
                        "id": canonical,
                        "embedding": emb,
                        "metadata": meta,
                        "document": doc,
                    })
                    to_delete.append(current_id)
                except Exception as e:
                    logger.error(f"Error preparing migration for id {current_id}: {e}")
                    error_count += 1

            if to_upsert:
                try:
                    collection.upsert(
                        ids=[item["id"] for item in to_upsert],
                        embeddings=[item["embedding"] for item in to_upsert],
                        metadatas=[item["metadata"] for item in to_upsert],
                        documents=[item["document"] for item in to_upsert],
                    )
                    migrated_count += len(to_upsert)
                except Exception as e:
                    logger.error(f"Error upserting migrated items: {e}")
                    error_count += len(to_upsert)

            if to_delete:
                try:
                    collection.delete(ids=to_delete)
                except Exception as e:
                    logger.error(f"Error deleting legacy ids: {e}")
                    # non-fatal

            if offset is None:
                break
            offset += len(ids)

        return {"migrated": migrated_count, "errors": error_count}
