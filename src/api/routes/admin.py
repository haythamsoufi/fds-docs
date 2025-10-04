"""Admin routes for system management."""

from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from src.core.database import get_db
from src.core.cache import cache
from src.core.config import settings

router = APIRouter()


class SystemStatsResponse(BaseModel):
    documents: Dict[str, Any]
    chunks: Dict[str, Any]
    queries: Dict[str, Any]
    cache: Dict[str, Any]


class PopulateVectorsResponse(BaseModel):
    message: str
    status: str
class ReembedRequest(BaseModel):
    document_id: str | None = None
    force: bool = False
    version: int | None = None


class ReindexRequest(BaseModel):
    document_id: str | None = None
    force_reprocess: bool = False
    clear_cache: bool = False


class CacheControlRequest(BaseModel):
    cache_type: str = "all"  # all, embeddings, queries, retrieval
    action: str = "clear"  # clear, stats, invalidate_pattern
    pattern: str | None = None
    version: int | None = None


@router.post("/reembed", response_model=PopulateVectorsResponse)
async def reembed_vectors(
    background_tasks: BackgroundTasks,
    request: Request,
    payload: ReembedRequest,
    db: AsyncSession = Depends(get_db)
) -> PopulateVectorsResponse:
    """Re-embed chunks for a document or all documents (if document_id not provided).

    - If version is provided, updates the embedding version used for cache keys.
    - If force is true, clears relevant embedding cache before re-embedding.
    """
    try:
        from ...services.embedding_service import EmbeddingService
        from ...core.models import ChunkModel
        from sqlalchemy import select

        # Optionally update embedding version at runtime
        if payload.version is not None:
            settings.embedding_version = payload.version

        embedding_service = getattr(request.app.state, "embedding_service", None)
        if not embedding_service:
            raise HTTPException(status_code=503, detail="Embedding service not available")

        # Clear caches if forced
        if payload.force:
            # Clear both query and passage caches for all versions or current version
            await embedding_service.clear_embeddings_cache()

        # Schedule background task to re-embed
        background_tasks.add_task(_reembed_task, embedding_service, db, payload.document_id)

        return PopulateVectorsResponse(
            message="Re-embedding started",
            status="processing"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@router.post("/reprocess-documents")
async def reprocess_all_documents(
    background_tasks: BackgroundTasks,
    request: Request,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Trigger reprocessing of all documents in the documents directory.

    This schedules the work as a background task and returns immediately.
    The UI can poll `/api/v1/documents/status/summary` for progress.
    """
    # If Celery/Redis enabled, enqueue task; else fallback to in-process background task
    try:
        if settings.use_redis:
            from ...services.tasks import process_documents_directory
            process_documents_directory.delay()
            return {
                "message": "Document reprocessing enqueued",
                "documents_path": settings.documents_path,
                "supported_formats": settings.supported_formats,
            }
    except Exception:
        pass

    processor = getattr(request.app.state, "document_processor", None)
    if not processor:
        raise HTTPException(status_code=503, detail="Document processor not available")
    background_tasks.add_task(processor.process_directory, settings.documents_path)

    return {
        "message": "Document reprocessing started",
        "documents_path": settings.documents_path,
        "supported_formats": settings.supported_formats,
    }


@router.post("/reindex", response_model=PopulateVectorsResponse)
async def reindex_documents(
    background_tasks: BackgroundTasks,
    request: Request,
    payload: ReindexRequest,
    db: AsyncSession = Depends(get_db)
) -> PopulateVectorsResponse:
    """Reindex documents with optional reprocessing and cache clearing."""
    try:
        processor = getattr(request.app.state, "document_processor", None)
        if not processor:
            raise HTTPException(status_code=503, detail="Document processor not available")

        # Clear cache if requested
        if payload.clear_cache:
            await clear_cache_route()

        # Schedule background task for reindexing
        background_tasks.add_task(
            _reindex_task, 
            processor, 
            db, 
            payload.document_id, 
            payload.force_reprocess
        )

        return PopulateVectorsResponse(
            message="Reindexing started",
            status="processing"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache-control")
async def cache_control(request: Request, payload: CacheControlRequest) -> Dict[str, Any]:
    """Advanced cache control operations."""
    try:
        from ...services.embedding_service import EmbeddingService
        
        embedding_service = getattr(request.app.state, "embedding_service", None)
        if not embedding_service:
            raise HTTPException(status_code=503, detail="Embedding service not available")

        result = {"action": payload.action, "cache_type": payload.cache_type}

        if payload.action == "clear":
            if payload.cache_type == "all":
                await cache.clear_all()
                result["message"] = "All caches cleared"
            elif payload.cache_type == "embeddings":
                await embedding_service.clear_embeddings_cache()
                result["message"] = "Embedding caches cleared"
            elif payload.cache_type == "queries":
                await cache.clear_pattern("retrieval:*")
                result["message"] = "Query caches cleared"
            elif payload.cache_type == "retrieval":
                await cache.clear_pattern("retrieval:*")
                result["message"] = "Retrieval caches cleared"
            else:
                raise HTTPException(status_code=400, detail="Invalid cache_type")

        elif payload.action == "stats":
            stats = await _get_cache_stats(payload.cache_type)
            result.update(stats)

        elif payload.action == "invalidate_pattern":
            if not payload.pattern:
                raise HTTPException(status_code=400, detail="Pattern required for invalidate_pattern")
            await cache.clear_pattern(payload.pattern)
            result["message"] = f"Cache entries matching '{payload.pattern}' cleared"

        else:
            raise HTTPException(status_code=400, detail="Invalid action")

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear-cache")
async def clear_cache_route() -> Dict[str, str]:
    """Clear all caches (no-op if Redis disabled)."""
    try:
        if getattr(cache, "redis", None):
            await cache.redis.flushdb()
        # Always succeed to avoid coupling UI to Redis
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset-vector-backpressure")
async def reset_vector_backpressure(request: Request) -> Dict[str, str]:
    """Reset vector database backpressure state."""
    try:
        embedding_service = getattr(request.app.state, "embedding_service", None)
        if not embedding_service:
            raise HTTPException(status_code=503, detail="Embedding service not available")
        
        # Reset backpressure state
        if hasattr(embedding_service, 'vector_store') and hasattr(embedding_service.vector_store, 'reset_backpressure_state'):
            embedding_service.vector_store.reset_backpressure_state()
            return {"message": "Vector database backpressure state reset successfully"}
        else:
            return {"message": "Vector store does not support backpressure reset"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/populate-vectors", response_model=PopulateVectorsResponse)
async def populate_vector_store(
    background_tasks: BackgroundTasks,
    request: Request,
    db: AsyncSession = Depends(get_db)
) -> PopulateVectorsResponse:
    """Populate the vector store with embeddings from existing chunks.
    
    This schedules the work as a background task and returns immediately.
    """
    try:
        from ...services.embedding_service import EmbeddingService, VectorStore
        from ...core.models import ChunkModel
        from sqlalchemy import select
        
        # If Celery/Redis enabled, enqueue task; else fallback
        try:
            if settings.use_redis:
                from ...services.tasks import embed_document_chunks
                embed_document_chunks.delay(None)
                return PopulateVectorsResponse(message="Vector population enqueued", status="processing")
        except Exception:
            pass

        embedding_service = getattr(request.app.state, "embedding_service", None)
        if not embedding_service:
            raise HTTPException(status_code=503, detail="Embedding service not available")
        background_tasks.add_task(_populate_vectors_task, embedding_service, db)
        
        return PopulateVectorsResponse(
            message="Vector store population started",
            status="processing"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset-vectors", response_model=PopulateVectorsResponse)
async def reset_vector_store(
    background_tasks: BackgroundTasks,
    request: Request,
    db: AsyncSession = Depends(get_db)
) -> PopulateVectorsResponse:
    """Delete the current vector collection and repopulate from DB chunks."""
    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        # Reset/delete collection
        chroma_client = chromadb.PersistentClient(
            path=settings.vector_db_path,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
        )

        try:
            collection = chroma_client.get_collection(name="document_chunks")
            collection.delete(where={})  # remove all items
        except Exception:
            # If not present, that's fine
            pass

        # Trigger fresh population
        embedding_service = getattr(request.app.state, "embedding_service", None)
        if not embedding_service:
            raise HTTPException(status_code=503, detail="Embedding service not available")

        background_tasks.add_task(_populate_vectors_task, embedding_service, db)

        return PopulateVectorsResponse(
            message="Vector store reset and repopulation started",
            status="processing"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _populate_vectors_task(embedding_service, db):
    """Background task to populate vector store."""
    try:
        from ...services.embedding_service import VectorStore
        from ...core.models import ChunkModel
        from sqlalchemy import select
        
        # Initialize vector store
        vector_store = VectorStore(embedding_service)
        
        # Get all chunks from database
        result = await db.execute(select(ChunkModel))
        chunks = result.scalars().all()
        
        if chunks:
            # Add chunks to vector store
            vector_ids = await vector_store.add_documents(chunks)
            
            # Update chunk records with embedding IDs
            updated_count = 0
            for chunk, vector_id in zip(chunks, vector_ids):
                if vector_id:
                    chunk.embedding_id = vector_id
                    updated_count += 1
                    
            await db.commit()
            
            print(f"Vector store populated: {updated_count} chunks processed")
        else:
            print("No chunks found to populate vector store")
            
    except Exception as e:
        print(f"Error populating vector store: {e}")


async def _reembed_task(embedding_service, db, document_id: str | None):
    """Background task to re-embed chunks and upsert into vector store."""
    try:
        from ...services.embedding_service import VectorStore
        from ...core.models import ChunkModel
        from sqlalchemy import select

        # Initialize vector store
        vector_store = VectorStore(embedding_service)

        # Select chunks to (re)embed
        if document_id:
            result = await db.execute(select(ChunkModel).where(ChunkModel.document_id == document_id))
        else:
            result = await db.execute(select(ChunkModel))
        chunks = result.scalars().all()

        if not chunks:
            print("No chunks found for re-embedding")
            return

        # Upsert into vector store
        vector_ids = await vector_store.add_documents(chunks)

        # Update chunk records with embedding IDs
        updated_count = 0
        for chunk, vector_id in zip(chunks, vector_ids):
            if vector_id:
                chunk.embedding_id = vector_id
                updated_count += 1
        await db.commit()

        print(f"Re-embedding completed: {updated_count} chunks updated")
    except Exception as e:
        print(f"Error during re-embedding: {e}")


class MigrateVectorsResponse(BaseModel):
    migrated: int
    errors: int


@router.post("/migrate-vectors", response_model=MigrateVectorsResponse)
async def migrate_vectors(
    request: Request,
):
    """Migrate existing vectors to new schema and consistent IDs."""
    try:
        from ...services.embedding_service import EmbeddingService, VectorStore

        embedding_service = getattr(request.app.state, "embedding_service", None)
        if not embedding_service:
            raise HTTPException(status_code=503, detail="Embedding service not available")

        vector_store = VectorStore(embedding_service)
        result = await vector_store.migrate_collection_ids()
        return MigrateVectorsResponse(migrated=result.get("migrated", 0), errors=result.get("errors", 0))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats(
    db: AsyncSession = Depends(get_db)
) -> SystemStatsResponse:
    """Get system statistics."""
    from sqlalchemy import select, func
    from ...core.models import DocumentModel, ChunkModel
    
    try:
        # Get document statistics
        result = await db.execute(
            select(
                DocumentModel.status,
                func.count(DocumentModel.id).label('count')
            ).group_by(DocumentModel.status)
        )
        status_counts = {row.status: row.count for row in result}
        
        total_documents = sum(status_counts.values())
        processed_documents = status_counts.get('completed', 0)
        failed_documents = status_counts.get('failed', 0)
        processing_documents = status_counts.get('processing', 0)
        pending_documents = status_counts.get('pending', 0)
        
        # Get chunk statistics
        chunk_result = await db.execute(select(func.count(ChunkModel.id)))
        total_chunks = chunk_result.scalar()
        
        # Get chunks with embeddings
        embedding_result = await db.execute(
            select(func.count(ChunkModel.id)).where(ChunkModel.embedding_id.isnot(None))
        )
        chunks_with_embeddings = embedding_result.scalar()
        
        # Get cache statistics
        cache_stats = await _get_cache_stats("all")
        
        return SystemStatsResponse(
            documents={
                "total": total_documents,
                "processed": processed_documents,
                "failed": failed_documents,
                "processing": processing_documents,
                "pending": pending_documents
            },
            chunks={
                "total": total_chunks,
                "with_embeddings": chunks_with_embeddings,
                "embedding_coverage": (chunks_with_embeddings / total_chunks * 100) if total_chunks > 0 else 0
            },
            queries={"total": 0, "avg_response_time": 0.0},
            cache=cache_stats
        )
    except Exception as e:
        return SystemStatsResponse(
            documents={"total": 0, "processed": 0, "failed": 0, "processing": 0, "pending": 0, "error": str(e)},
            chunks={"total": 0, "with_embeddings": 0, "embedding_coverage": 0},
            queries={"total": 0, "avg_response_time": 0.0},
            cache={"hit_rate": 0.0, "total_keys": 0}
        )


async def _reindex_task(processor, db, document_id: str | None, force_reprocess: bool):
    """Background task to reindex documents."""
    try:
        if document_id:
            # Reindex specific document
            if force_reprocess:
                # Reprocess document file
                document = await db.get(DocumentModel, document_id)
                if document:
                    await processor.process_document(document.file_path)
            else:
                # Just re-embed existing chunks
                from ...services.embedding_service import EmbeddingService, VectorStore
                from sqlalchemy import select
                
                embedding_service = EmbeddingService()
                vector_store = VectorStore(embedding_service)
                
                result = await db.execute(
                    select(ChunkModel).where(ChunkModel.document_id == document_id)
                )
                chunks = result.scalars().all()
                
                if chunks:
                    vector_ids = await vector_store.add_documents(chunks)
                    for chunk, vector_id in zip(chunks, vector_ids):
                        if vector_id:
                            chunk.embedding_id = vector_id
                    await db.commit()
                    print(f"Reindexed {len(chunks)} chunks for document {document_id}")
        else:
            # Reindex all documents
            if force_reprocess:
                await processor.process_directory(settings.documents_path)
            else:
                # Re-embed all existing chunks
                from ...services.embedding_service import EmbeddingService, VectorStore
                from sqlalchemy import select
                
                embedding_service = EmbeddingService()
                vector_store = VectorStore(embedding_service)
                
                result = await db.execute(select(ChunkModel))
                chunks = result.scalars().all()
                
                if chunks:
                    vector_ids = await vector_store.add_documents(chunks)
                    for chunk, vector_id in zip(chunks, vector_ids):
                        if vector_id:
                            chunk.embedding_id = vector_id
                    await db.commit()
                    print(f"Reindexed {len(chunks)} chunks")
                    
    except Exception as e:
        print(f"Error during reindexing: {e}")


async def _get_cache_stats(cache_type: str) -> Dict[str, Any]:
    """Get cache statistics."""
    try:
        if not getattr(cache, "redis", None):
            return {"hit_rate": 0.0, "total_keys": 0, "available": False}
        
        redis = cache.redis
        total_keys = await redis.dbsize()
        
        # Get cache hit/miss stats if available
        info = await redis.info("stats")
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0.0
        
        # Get specific cache type stats
        if cache_type == "embeddings":
            embedding_keys = len(await redis.keys("embedding:*"))
            return {
                "hit_rate": hit_rate,
                "total_keys": total_keys,
                "embedding_keys": embedding_keys,
                "available": True
            }
        elif cache_type == "retrieval":
            retrieval_keys = len(await redis.keys("retrieval:*"))
            return {
                "hit_rate": hit_rate,
                "total_keys": total_keys,
                "retrieval_keys": retrieval_keys,
                "available": True
            }
        else:
            return {
                "hit_rate": hit_rate,
                "total_keys": total_keys,
                "available": True
            }
    except Exception as e:
        return {"hit_rate": 0.0, "total_keys": 0, "available": False, "error": str(e)}
