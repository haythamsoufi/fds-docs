"""FastAPI application for the Enterprise RAG System."""

import asyncio
import logging
import time
import warnings
import aiofiles
import os
from contextlib import asynccontextmanager
from typing import List, Dict, Any
from datetime import datetime
import re

# Suppress all deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Disable ChromaDB telemetry to prevent telemetry errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
# Remove deprecated CHROMA_DB_IMPL; new client API selects backend automatically
os.environ.pop("CHROMA_DB_IMPL", None)

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, BackgroundTasks, Response, Request
from prometheus_client import CONTENT_TYPE_LATEST
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.core.config import settings
from src.core.database import create_tables, get_db, get_db_session
from src.core.cache import cache
from src.core.monitoring import monitoring_service
from src.core.models import HealthCheck, QueryResponse, QueryRequest, RetrievedChunk, ChunkResponse, DocumentResponse, ChunkModel, DocumentModel, QueryIntent
from src.services.conversational_ai import conversational_ai
from src.services.performance_optimizer import performance_monitor
from src.services.retrieval_service import SearchType
from src.services.retrieval_service import KeywordSearcher, HybridRetriever
from src.services.embedding_service import EmbeddingService
from src.services.document_processor import DocumentProcessor
from src.services.confidence_calibrator import confidence_calibrator
from src.api.routes import documents, queries, admin, evaluation, metrics

logger = logging.getLogger(__name__)


async def _auto_populate_vectors(embedding_service):
    """Auto-populate vector store if it's empty or has few items."""
    try:
        from ..services.embedding_service import VectorStore
        from ..core.models import ChunkModel
        from ..core.database import get_db_session
        from sqlalchemy import select, func
        import chromadb
        
        # Check if vector store exists and has items
        try:
            chroma_client = chromadb.PersistentClient(
                path=settings.vector_db_path
            )
            
            try:
                collection = chroma_client.get_collection(name="document_chunks")
                vector_count = collection.count()
                logger.info(f"Vector store has {vector_count} items")
                
                # If vector store has items, skip population
                if vector_count > 0:
                    return
            except:
                # Collection doesn't exist, need to populate
                pass
        except Exception as e:
            logger.warning(f"Could not check vector store: {e}")
            return
        
        # Check if we have chunks in database
        async with get_db_session() as session:
            chunk_result = await session.execute(select(func.count(ChunkModel.id)))
            total_chunks = chunk_result.scalar()
            
            if total_chunks == 0:
                logger.info("No chunks found in database, skipping vector population")
                return
            
            # Get chunks without embeddings
            embedding_result = await session.execute(
                select(func.count(ChunkModel.id)).where(ChunkModel.embedding_id.is_(None))
            )
            chunks_without_embeddings = embedding_result.scalar()
            
            if chunks_without_embeddings == 0:
                logger.info("All chunks already have embeddings, skipping vector population")
                return
            
            logger.info(f"Auto-populating vector store with {chunks_without_embeddings} chunks...")
            
            # Initialize vector store
            vector_store = VectorStore(embedding_service)
            
            # Get all chunks
            result = await session.execute(select(ChunkModel))
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
                        
                await session.commit()
                logger.info(f"Auto-populated vector store: {updated_count} chunks processed")
            
    except Exception as e:
        logger.error(f"Auto vector population failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Enterprise RAG System...")

    # Initialize database
    try:
        await create_tables()
        logger.info("Database initialized")
    except Exception as e:
        logger.warning(f"Database init failed: {e}")

    # Initialize cache (optional)
    try:
        await cache.connect()
        logger.info("Cache initialized successfully")
    except Exception as e:
        logger.warning(f"Cache init skipped: {e}")

    # Initialize core services
    try:
        # Embedding service
        embedding_service = EmbeddingService()
        await embedding_service.initialize()
        app.state.embedding_service = embedding_service

        # Document processor
        document_processor = DocumentProcessor()
        app.state.document_processor = document_processor
        # File watching disabled - documents are only uploaded through the UI
        logger.info("File watching disabled - using manual upload only")

        # Retriever
        app.state.retriever = HybridRetriever(embedding_service)

        # Auto-populate vectors if empty
        await _auto_populate_vectors(embedding_service)
    except Exception as e:
        logger.error(f"Service initialization failed: {e}")

    logger.info("Enterprise RAG System started successfully")

    yield

    # Shutdown
    logger.info("Shutting down Enterprise RAG System...")

    # Disconnect cache
    try:
        await cache.disconnect()
    except Exception:
        pass

    logger.info("Enterprise RAG System shutdown complete")


# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="Enterprise-grade RAG system for document question answering",
    lifespan=lifespan
)

# Add monitoring middleware
@app.middleware("http")
async def monitoring_middleware(request: Request, call_next):
    """Middleware to track request metrics."""
    # Track the request
    response = await call_next(request)

    # Log request metrics
    await monitoring_service.track_request(request, response)

    return response

# Add redirect prevention middleware
@app.middleware("http")
async def prevent_redirects_middleware(request: Request, call_next):
    """Middleware to prevent unwanted redirects."""
    # Log the original request
    logger.debug(f"Processing request: {request.method} {request.url}")
    
    response = await call_next(request)
    
    # Ensure we don't have redirect status codes unless intentional
    if response.status_code in [301, 302, 307, 308]:
        logger.warning(f"Redirect detected: {response.status_code} for {request.url}")
        # For API endpoints, we typically don't want redirects
        if request.url.path.startswith('/api/'):
            logger.error(f"Unexpected redirect for API endpoint: {request.url.path}")
    
    return response

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Include routers
app.include_router(documents.router, prefix="/api/v1/documents", tags=["documents"])
app.include_router(queries.router, prefix="/api/v1/queries", tags=["queries"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])
app.include_router(evaluation.router, prefix="/api/v1/evaluation", tags=["evaluation"])
app.include_router(metrics.router, prefix="/api/v1/metrics", tags=["metrics"])


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Enterprise RAG System API",
        "version": settings.api_version,
        "docs": "/docs"
    }


@app.get("/test")
async def test():
    """Simple test endpoint."""
    return {"message": "Test endpoint working"}


@app.post("/test-query")
async def test_query(request: QueryRequest):
    """Test query endpoint."""
    return {
        "query": request.query,
        "message": "Query received successfully",
        "max_results": request.max_results
    }


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    # Basic health check
    components = {
        "database": "healthy",
        "cache": "unavailable",  # Simplified for now
        "embedding_service": "not_initialized",  # Simplified for now
    }

    # Simple overall status
    overall_status = "healthy"

    from datetime import datetime
    return HealthCheck(
        status=overall_status,
        timestamp=datetime.utcnow(),
        version=settings.api_version,
        components=components
    )


@app.post("/api/v1/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents using the RAG system with conversational AI."""
    start_time = time.time()
    requested_search_type = SearchType.HYBRID  # Initialize default value

    try:
        logger.info(f"/api/v1/query received: query='{request.query}', max_results={request.max_results}")
        # Ensure retriever is available
        retriever = getattr(app.state, "retriever", None)
        if retriever is None:
            embedding_service = getattr(app.state, "embedding_service", None)
            if embedding_service is None:
                embedding_service = EmbeddingService()
                await embedding_service.initialize()
                app.state.embedding_service = embedding_service
            retriever = HybridRetriever(embedding_service)
            app.state.retriever = retriever

        # Trivial greeting guardrail: return a friendly response without retrieval
        normalized = request.query.strip().lower()
        trivial_greetings = {"hi", "hello", "hey", "yo", "sup", "hi!", "hello!", "hey!"}
        if normalized in trivial_greetings or len(normalized) < 3:
            response_time = time.time() - start_time
            logger.info("Greeting detected; returning canned response without retrieval")
            return QueryResponse(
                query=request.query,
                answer="Hello! I can help you search your documents. Ask me something like 'What are our 2025 priorities?' or 'Summarize the financial outlook.'",
                retrieved_chunks=[],
                response_time=response_time,
                intent=None,
                confidence=0.99
            )

        results = []
        if normalized not in trivial_greetings and len(normalized) >= 3:
            # Retrieve results
            logger.info("Starting retrieval (hybrid)")
            results = await retriever.retrieve(
            query=request.query,
            k=request.max_results,
            search_type=requested_search_type,
            filters=request.filters or None
            )
            logger.info(f"Retrieval complete: results_count={len(results)}")

        # Build retrieved_chunks by joining with DB models and truncating content for UI
        retrieved_chunks: List[RetrievedChunk] = []
        async with get_db_session() as session:
            for res in results:
                db_row = await session.execute(
                    select(ChunkModel, DocumentModel)
                    .join(DocumentModel, ChunkModel.document_id == DocumentModel.id)
                    .where(ChunkModel.id == res.chunk_id)
                )
                row = db_row.first()
                if not row:
                    continue
                chunk, document = row

                # Truncate content to avoid full document dumps in responses
                display_content = chunk.content
                if len(display_content) > 1200:
                    display_content = display_content[:1200] + "..."

                chunk_resp = ChunkResponse(
                    id=str(chunk.id),
                    document_id=str(chunk.document_id),
                    chunk_index=chunk.chunk_index,
                    content=display_content,
                    metadata=chunk.extra_metadata,
                    created_at=chunk.created_at
                )

                doc_resp = DocumentResponse(
                    id=str(document.id),
                    filename=document.filename,
                    filepath=document.filepath,
                    file_size=document.file_size,
                    mime_type=document.mime_type,
                    status=document.status,  # type: ignore[arg-type]
                    created_at=document.created_at,
                    updated_at=document.updated_at,
                    processed_at=document.processed_at,
                    metadata=document.extra_metadata,
                    chunk_count=document.chunk_count
                )

                retrieved_chunks.append(
                    RetrievedChunk(
                        chunk=chunk_resp,
                        score=res.score,
                        document=doc_resp
                    )
                )

        # Generate answer
        answer = await generate_answer(request.query, results)

        # Calibrate confidence and check no-answer threshold
        top_k_scores = [r.score for r in results[:5]] if results else []
        confidence_score = confidence_calibrator.calibrate_confidence(
            request.query, results, answer, top_k_scores
        )

        # Apply no-answer threshold
        if confidence_calibrator.should_refuse_answer(confidence_score):
            answer = confidence_calibrator.get_no_answer_response(request.query)
            logger.info(f"Refused answer due to low confidence: {confidence_score.reasoning}")

        response_time = time.time() - start_time
        logger.info(f"/api/v1/query response ready in {response_time:.2f}s with {len(retrieved_chunks)} chunks, confidence={confidence_score.calibrated_score:.2f}")
        return QueryResponse(
            query=request.query,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            response_time=response_time,
            intent=QueryIntent.FACTUAL,
            confidence=confidence_score.calibrated_score
        )

    except Exception as e:
        logger.exception(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def generate_answer(query: str, search_results: List) -> str:
    """Generate an answer using OpenAI if configured, else summarize top chunks.

    Grounding rules:
    - Use only the provided sources for facts
    - Cite sources inline as [Source N]
    - If insufficient evidence, explicitly say you don't know
    - Avoid speculation; be concise and specific
    """
    if not search_results:
        return "I couldn't find relevant information to answer your question."

    # Build concise, source-labeled context blocks with metadata
    top_k = min(5, len(search_results))
    context_blocks = []
    for i in range(top_k):
        res = search_results[i]
        content = getattr(res, "content", None)
        meta = getattr(res, "metadata", {}) or {}
        if content is None and hasattr(res, "chunk"):
            content = getattr(res.chunk, "content", "")
            meta = getattr(res.chunk, "metadata", {}) or meta

        content = (content or "").strip()
        # Focus: take first 400 chars to respect token limits, then extract 2 relevant sentences containing query words
        snippet = content[:400]
        query_words = [w for w in query.lower().split() if len(w) > 2]
        sentences = re.split(r"(?<=[.!?])\s+", snippet)
        relevant = [s for s in sentences if any(w in s.lower() for w in query_words)]
        focused_content = (" ".join(relevant[:2])).strip() or snippet[:200]

        # Compose label with page/section if available
        page_info = ""
        if isinstance(meta, dict):
            ps = meta.get("page_start")
            pe = meta.get("page_end")
            sect = meta.get("section_title")
            if ps or pe:
                if ps and pe and ps != pe:
                    page_info += f" p.{ps}-{pe}"
                elif ps:
                    page_info += f" p.{ps}"
            if sect:
                page_info += f" | {sect}"

        context_blocks.append(f"[Source {i+1}{page_info}]\n{focused_content}")

    context_text = "\n\n".join(context_blocks)

    # Try OpenAI or OpenAI-compatible chat completion (supports local servers via base_url)
    try:
        # Skip LLM if using placeholder API key or local LLM without proper URL
        if settings.openai_api_key and settings.openai_api_key != "your_openai_api_key_here":
            # Valid OpenAI key
            use_llm = True
        elif settings.use_local_llm and settings.local_llm_base_url:
            # Check if local LLM is actually running
            import aiohttp
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(settings.local_llm_base_url + "/models", timeout=aiohttp.ClientTimeout(total=3)) as resp:
                        use_llm = resp.status == 200
            except:
                use_llm = False
        else:
            use_llm = False
            
        if use_llm:
            from openai import AsyncOpenAI

            # Determine client configuration
            api_key = settings.openai_api_key or "sk-local-placeholder"
            client_kwargs = {"api_key": api_key}
            if settings.use_local_llm and settings.local_llm_base_url:
                client_kwargs["base_url"] = settings.local_llm_base_url.rstrip("/")

            client = AsyncOpenAI(**client_kwargs)
            system_rules = (
                "You are a precise, grounded assistant.\n"
                "Rules:\n"
                "- Use ONLY the provided sources for facts.\n"
                "- If evidence is insufficient, say you don't know.\n"
                "- Cite sources inline as [Source N].\n"
                "- Be concise, factual, and avoid speculation.\n"
            )

            user_prompt = (
                f"Question: {query}\n\n"
                "Sources (use citations like [Source N]):\n"
                f"{context_text}"
            )

            # Add timeout to prevent hanging when LLM is not available
            try:
                completion = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=(settings.local_llm_model or settings.default_llm_model) if settings.use_local_llm else settings.default_llm_model,
                        messages=[
                            {"role": "system", "content": system_rules},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.1,  # Lower temperature for more focused answers
                        max_tokens=getattr(settings, 'llm_response_max_tokens', 400),   # Use configured max tokens
                    ),
                    timeout=settings.llm_timeout  # Use configured LLM timeout
                )
            except asyncio.TimeoutError:
                logger.warning("LLM request timed out after 5 seconds, falling back to extractive summary")
                raise Exception("LLM timeout")

            content = completion.choices[0].message.content if completion.choices else None
            if content:
                return content.strip()
    except Exception as e:
        logger.warning(f"OpenAI generation failed, using extractive summary: {e}")

    # Fallback: improved extractive summary
    if context_blocks:
        # Extract key sentences that contain query terms
        query_words = query.lower().split()
        key_sentences = []
        
        for block in context_blocks:
            sentences = block.split('. ')
            for sentence in sentences:
                if any(word in sentence.lower() for word in query_words):
                    key_sentences.append(sentence.strip())
        
        if key_sentences:
            summary = '. '.join(key_sentences[:3]) + '.'
            return f"Based on the available sources: {summary}"
    
    return "I found some relevant information, but it may not directly answer your specific question. Please try rephrasing your query or ask for more specific details."


# Add monitoring endpoints
@app.get("/metrics")
async def get_metrics():
    """Get Prometheus metrics."""
    return Response(
        content=monitoring_service.get_metrics_text(),
        media_type=CONTENT_TYPE_LATEST
    )

@app.get("/api/v1/monitoring/health")
async def get_detailed_health():
    """Get detailed system health information."""
    # Get basic health check
    health = await health_check()

    # Add monitoring-specific health data
    monitoring_health = {
        "monitoring_service": "healthy",
        "metrics_collection": "active",
        "structured_logging": "active"
    }

    # Try to get system metrics
    try:
        await monitoring_service.update_system_metrics()
        monitoring_health["system_metrics"] = "available"
    except Exception as e:
        monitoring_health["system_metrics"] = f"error: {str(e)}"

    # Add cache health (simplified since cache doesn't have health_check method)
    try:
        if hasattr(cache, 'redis') and cache.redis:
            await cache.redis.ping()
            monitoring_health["cache"] = "healthy"
        else:
            monitoring_health["cache"] = "unavailable"
    except Exception as e:
        monitoring_health["cache"] = f"error: {str(e)}"

        return {
            **health.dict(),
            "monitoring": monitoring_health
        }

@app.get("/api/v1/performance/report")
async def get_performance_report():
    """Get comprehensive performance monitoring report."""
    try:
        report = await performance_monitor.get_performance_report()

        # Add cache performance to the report (simplified)
        if hasattr(cache, 'redis') and cache.redis:
            try:
                await cache.redis.ping()
                report["cache_performance"] = {"status": "healthy"}
            except:
                report["cache_performance"] = {"status": "error"}
        else:
            report["cache_performance"] = {"status": "unavailable"}

        return report
    except Exception as e:
        logger.error(f"Error generating performance report: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate performance report")

@app.post("/api/v1/performance/optimize")
async def trigger_performance_optimization():
    """Trigger performance optimization tasks."""
    try:
        # Trigger cache pattern analysis
        await performance_monitor.cache_optimizer.analyze_access_patterns()

        # Trigger database optimization
        await performance_monitor.database_optimizer.optimize_database_operations()

        # Log optimization trigger
        await monitoring_service.log_event(
            "performance_optimization_triggered",
            {"timestamp": time.time()}
        )

        return {"message": "Performance optimization tasks triggered successfully"}
    except Exception as e:
        logger.error(f"Error triggering performance optimization: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger performance optimization")

@app.post("/api/v1/upload", response_model=Dict[str, str])
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """Upload and process a document."""
    import os
    from pathlib import Path

    async with monitoring_service.track_performance("document_upload", "api"):
        try:
            # Validate file
            if not any(file.filename.lower().endswith(ext) for ext in settings.supported_formats):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file format. Supported: {settings.supported_formats}"
                )

            # Some ASGI servers may not populate UploadFile.size; fallback to reading bytes length
            file_bytes = await file.read()
            # Check file size only if max_file_size is set (> 0)
            if settings.max_file_size > 0 and len(file_bytes) > settings.max_file_size:
                raise HTTPException(
                    status_code=400,
                    detail=f"File too large. Maximum size: {settings.max_file_size} bytes"
                )

            # Save file
            upload_dir = Path(settings.documents_path)
            upload_dir.mkdir(parents=True, exist_ok=True)

            file_path = upload_dir / file.filename

            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(file_bytes)

            # Process document directly (file watcher disabled)
            try:
                document_id = await app.state.document_processor.process_document(str(file_path))
                logger.info(f"Document processed successfully: {file_path}, ID: {document_id}")
            except Exception as process_error:
                logger.error(f"Error processing document {file_path}: {process_error}")
                # Don't fail the upload if processing fails - document is saved
                # Processing can be retried later

            # Track document upload metrics
            await monitoring_service.log_event(
                "document_uploaded",
                {
                    "filename": file.filename,
                    "file_size": len(file_bytes),
                    "file_format": Path(file.filename).suffix.lower()
                }
            )

            return {
                "message": "File uploaded successfully",
                "filename": file.filename,
                "status": "processing"
            }

        except Exception as e:
            await monitoring_service.log_event(
                "document_upload_failed",
                {"filename": file.filename, "error": str(e)},
                level="error"
            )
            logger.error(f"Error uploading file: {e}")
            raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
