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
from src.services.retrieval_service import SearchType, SearchResult
from src.services.retrieval_service import KeywordSearcher, HybridRetriever
from src.services.enhanced_retrieval_service import EnhancedRetriever
from src.services.embedding_service import EmbeddingService
from src.services.document_processor import DocumentProcessor
from src.services.multimodal_document_processor import multimodal_document_processor
from src.services.confidence_calibrator import confidence_calibrator
from src.api.routes import documents, queries, admin, evaluation, metrics, docs

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

        # Enhanced Retriever (supports structured data)
        app.state.retriever = HybridRetriever(embedding_service)
        app.state.enhanced_retriever = EnhancedRetriever(embedding_service)

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
app.include_router(docs.router, tags=["documentation"])


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
    # Determine requested search type (UI can pass via filters.search_type)
    requested_search_type = SearchType.HYBRID  # default
    try:
        # Prefer explicit request filter, else use configured default
        requested_type_str = None
        if isinstance(request.filters, dict):
            requested_type_str = request.filters.get("search_type")
        if not requested_type_str:
            requested_type_str = getattr(settings, "search_type", "hybrid")
        requested_type_str = str(requested_type_str).lower().strip()
        if requested_type_str in {"semantic", "keyword", "hybrid"}:
            requested_search_type = SearchType(requested_type_str)
    except Exception:
        # Fallback to default HYBRID if anything goes wrong
        requested_search_type = SearchType.HYBRID

    try:
        logger.info(f"/api/v1/query received: query='{request.query}', max_results={request.max_results}")
        # Ensure enhanced retriever is available
        enhanced_retriever = getattr(app.state, "enhanced_retriever", None)
        if enhanced_retriever is None:
            embedding_service = getattr(app.state, "embedding_service", None)
            if embedding_service is None:
                embedding_service = EmbeddingService()
                await embedding_service.initialize()
                app.state.embedding_service = embedding_service
            enhanced_retriever = EnhancedRetriever(embedding_service)
            app.state.enhanced_retriever = enhanced_retriever

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
            # Retrieve results with enhanced retrieval (text + structured data)
            logger.info(f"Starting enhanced retrieval ({requested_search_type.value})")
            # Determine k (max results) dynamically based on query characteristics
            base_k = settings.retrieval_k  # Now defaults to 100
            word_count = len(normalized.split())
            is_numeric_q = bool(re.search(r"\b(how many|how much|number of|count of|percentage|percent|total)\b", normalized))
            # Increase recall for longer/analytical or numeric queries
            if word_count >= 16:
                base_k = max(base_k, 150)  # Very high for complex queries
            elif word_count >= 8:
                base_k = max(base_k, 120)  # High for medium queries
            if is_numeric_q:
                base_k = max(base_k, 140)  # Very high for numeric queries
            user_k = request.max_results if request.max_results and request.max_results > 0 else None
            k_value = max(min(user_k or base_k, 200), 10)  # Cap at 200, minimum 10
            text_results, structured_results = await enhanced_retriever.retrieve_enhanced(
                query=request.query,
                k=k_value,
                search_type=requested_search_type,
                filters=request.filters or None,
                include_structured_data=True
            )
            # Fallback strategy: if no text results, try semantic then keyword
            if not text_results and requested_search_type == SearchType.HYBRID:
                logger.info("Hybrid returned 0 text results; retrying with semantic-only")
                text_results, _ = await enhanced_retriever.retrieve_enhanced(
                    query=request.query,
                    k=k_value,
                    search_type=SearchType.SEMANTIC,
                    filters={k: v for k, v in (request.filters or {}).items() if k != "search_type"},
                    include_structured_data=True
                )
                if not text_results:
                    logger.info("Semantic returned 0 text results; retrying with keyword-only")
                    text_results, _ = await enhanced_retriever.retrieve_enhanced(
                        query=request.query,
                        k=k_value,
                        search_type=SearchType.KEYWORD,
                        filters={k: v for k, v in (request.filters or {}).items() if k != "search_type"},
                        include_structured_data=True
                    )
            logger.info(f"Enhanced retrieval complete: text_results={len(text_results)}, structured_results={len(structured_results)}")

        # Build retrieved_chunks by joining with DB models and truncating content for UI
        retrieved_chunks: List[RetrievedChunk] = []
        async with get_db_session() as session:
            for res in text_results:
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

        # Numeric strict path: try to answer "how many ... in YEAR" from structured data
        strict_numeric_answer: Optional[str] = None
        year_match = re.search(r"\b(20\d{2}|19\d{2})\b", normalized)
        is_how_many = bool(re.search(r"\b(how many|number of|count of)\b", normalized))
        if is_how_many and year_match and structured_results:
            target_year = year_match.group(1)
            # Look for numeric structured entries related to operations in the target year
            for s in structured_results:
                data = s.get("data", {}) or {}
                text_repr = data.get("text_representation", "")
                if target_year in text_repr.lower() or target_year in (s.get("metadata", {}) or {}).get("text", "").lower():
                    # Extract first plausible integer
                    m = re.search(r"\b(\d{1,6})\b", text_repr)
                    if m:
                        strict_numeric_answer = m.group(1)
                        break

        # Generate answer (LLM/extractive), preferring numeric strict answer when present
        answer = await generate_answer_enhanced(request.query, text_results, structured_results)
        if strict_numeric_answer:
            # Prepend a concise numeric statement and keep the rest as supporting context
            answer = f"{strict_numeric_answer} [Source 1]\n\n" + answer

        # Build citations from top retrieved chunks (limit 3)
        citations: List[Dict[str, Any]] = []
        for rc in retrieved_chunks[:3]:
            citations.append({
                "id": rc.chunk.id,
                "document_id": rc.document.id,
                "document_title": (rc.document.metadata or {}).get("title") if rc.document.metadata else rc.document.id,
                "page_number": (rc.chunk.metadata or {}).get("page_number") if rc.chunk.metadata else None,
                "section_title": (rc.chunk.metadata or {}).get("section_title") if rc.chunk.metadata else None,
                "chunk_id": rc.chunk.id,
                "score": rc.score,
                "content": rc.chunk.content,
                "metadata": rc.chunk.metadata or {}
            })

        # Calibrate confidence and check no-answer threshold
        top_k_scores = [r.score for r in text_results[:5]] if text_results else []
        confidence_score = confidence_calibrator.calibrate_confidence(
            request.query, text_results, answer, top_k_scores
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
            confidence=confidence_score.calibrated_score,
            citations=citations
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
            # Add numeric QA guidance for count/how many questions
            numeric_hint = """
If the question asks for a count or number (e.g., "how many", "number of"), extract the exact number from the sources. If multiple numbers exist, prefer the most recent and clearly labeled figure. If uncertain, answer with the best supported range and cite.
""".strip()

            system_rules = (
                "You are a precise, grounded assistant.\n"
                "Rules:\n"
                "- Use ONLY the provided sources for facts.\n"
                "- If evidence is insufficient, say you don't know.\n"
                "- Cite sources inline as [Source N].\n"
                "- Be concise, factual, and avoid speculation.\n"
                f"\n{numeric_hint}"
            )

            user_prompt = (
                f"Question: {query}\n\n"
                "Sources (use citations like [Source N]):\n"
                f"{context_text}"
            )

            # Add timeout to prevent hanging when LLM is not available
            try:
                create_kwargs = {
                    "model": (settings.local_llm_model or settings.default_llm_model) if settings.use_local_llm else settings.default_llm_model,
                    "messages": [
                        {"role": "system", "content": system_rules},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": getattr(settings, 'llm_temperature', 0.1),
                }
                if getattr(settings, 'llm_response_max_tokens', 0) and settings.llm_response_max_tokens > 0:
                    create_kwargs["max_tokens"] = settings.llm_response_max_tokens
                create_call = client.chat.completions.create(**create_kwargs)
                if settings.llm_timeout and settings.llm_timeout > 0:
                    completion = await asyncio.wait_for(create_call, timeout=settings.llm_timeout)
                else:
                    completion = await create_call
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


async def generate_answer_enhanced(query: str, text_results: List[SearchResult], structured_results: List[Dict[str, Any]]) -> str:
    """Generate an enhanced answer using both text and structured data sources."""
    
    if not text_results and not structured_results:
        return "I don't have enough information to answer your question based on the available sources."
    
    # Build context from text results
    context_blocks = []
    for i, result in enumerate(text_results[:5], 1):
        # Use the content directly
        content = result.content
        context_blocks.append(f"[Source {i}] {content}")
    
    # Build context from structured data
    structured_context = []
    for i, struct_data in enumerate(structured_results[:3], 1):
        content_type = struct_data.get("content_type", "data")
        data = struct_data.get("data", {})
        
        if content_type == "table":
            # Add table summary
            table_text = data.get("text_representation", "")
            if table_text:
                structured_context.append(f"[Table Data {i}] {table_text}")
        
        elif content_type == "chart":
            # Add chart summary
            chart_text = data.get("text_representation", "")
            if chart_text:
                structured_context.append(f"[Chart Data {i}] {chart_text}")
        
        elif content_type == "numeric":
            # Add numeric data
            numeric_data = data
            value = numeric_data.get("value")
            original_text = numeric_data.get("original_text", "")
            context = numeric_data.get("context", {})
            
            if value and original_text:
                context_info = ""
                if context.get("headers"):
                    context_info = f" (from table columns: {', '.join(context['headers'])})"
                structured_context.append(f"[Numeric Data {i}] {original_text}{context_info}")
    
    # Combine all context
    all_context = context_blocks + structured_context
    context_text = "\n\n".join(all_context)

    # Lightweight verification helper: keep only sentences supported by context
    def _verify_and_prune(answer_text: str, context_text: str) -> str:
        try:
            if not answer_text:
                return answer_text
            ctx_lower = context_text.lower()
            sentences = re.split(r"(?<=[.!?])\s+", answer_text)
            kept = []
            for s in sentences:
                s_clean = s.strip()
                if not s_clean:
                    continue
                # Keep sentence if key tokens appear in context
                tokens = [t for t in re.findall(r"[a-z0-9%]+", s_clean.lower()) if len(t) > 3]
                if not tokens:
                    continue
                overlap = sum(1 for t in tokens if t in ctx_lower)
                if overlap >= max(2, len(tokens)//6):
                    kept.append(s_clean)
            if kept:
                return " ".join(kept)
            return answer_text
        except Exception:
            return answer_text
    
    # Check if we should use LLM or fallback to extractive summary
    if not settings.use_local_llm or not settings.local_llm_base_url:
        # Fallback to extractive summary
        return await _extractive_summary_with_structured_data(query, text_results, structured_results)
    
    try:
        from openai import AsyncOpenAI
        
        # Use LLM with enhanced context
        api_key = settings.openai_api_key or "sk-local-placeholder"
        client_kwargs = {"api_key": api_key}
        if settings.use_local_llm and settings.local_llm_base_url:
            client_kwargs["base_url"] = settings.local_llm_base_url.rstrip("/")

        client = AsyncOpenAI(**client_kwargs)
        
        # Enhanced system rules for multimodal data
        system_rules = (
            "You are a precise, grounded assistant that can analyze both text and structured data (tables, charts, numeric values).\n"
            "Rules:\n"
            "- Use ONLY the provided sources for facts.\n"
            "- Pay special attention to numeric data from tables and charts.\n"
            "- If the question asks for specific numbers, extract them from structured data sources.\n"
            "- If evidence is insufficient, say you don't know.\n"
            "- Cite sources inline as [Source N], [Table Data N], [Chart Data N], or [Numeric Data N].\n"
            "- Be concise, factual, and avoid speculation.\n"
            "- For numeric queries, prefer exact values from structured data over text descriptions."
        )

        user_prompt = (
            f"Question: {query}\n\n"
            "Sources (use appropriate citations):\n"
            f"{context_text}"
        )

        # Add timeout to prevent hanging
        try:
            create_kwargs = {
                "model": settings.local_llm_model or "llama3.1:70b",
                "messages": [
                    {"role": "system", "content": system_rules},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": settings.llm_temperature,
            }
            if getattr(settings, 'llm_response_max_tokens', 0) and settings.llm_response_max_tokens > 0:
                create_kwargs["max_tokens"] = settings.llm_response_max_tokens
            create_call = client.chat.completions.create(**create_kwargs)
            if settings.llm_timeout and settings.llm_timeout > 0:
                completion = await asyncio.wait_for(create_call, timeout=settings.llm_timeout)
            else:
                completion = await create_call
            
            answer = completion.choices[0].message.content.strip()
            # Verification pass: prune unsupported sentences
            verified = _verify_and_prune(answer, context_text)
            return verified
            
        except asyncio.TimeoutError:
            logger.warning("LLM request timed out, falling back to extractive summary")
            return await _extractive_summary_with_structured_data(query, text_results, structured_results)
            
    except Exception as e:
        logger.warning(f"LLM generation failed: {e}, falling back to extractive summary")
        return await _extractive_summary_with_structured_data(query, text_results, structured_results)


async def _extractive_summary_with_structured_data(query: str, text_results: List[SearchResult], structured_results: List[Dict[str, Any]]) -> str:
    """Create an extractive summary using both text and structured data."""
    
    if not text_results and not structured_results:
        return "I don't have enough information to answer your question."
    
    # Extract key information from text results with simple relevance scoring
    key_sentences: List[str] = []
    query_words = [word.lower() for word in re.findall(r"[a-zA-Z0-9%]+", query) if len(word) > 2]
    
    for result in text_results[:10]:
        content = (result.content or "").strip()
        if not content:
            continue
        # Split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", content)
        # Score sentences by overlap with query words
        scored = []
        for s in sentences:
            s_lower = s.lower()
            score = sum(1 for w in query_words if w in s_lower)
            if score > 0:
                scored.append((score, s.strip()))
        # Keep top 2 per chunk
        scored.sort(key=lambda x: x[0], reverse=True)
        key_sentences.extend([s for _, s in scored[:2]])
    
    # Extract key information from structured data
    structured_info = []
    for struct_data in structured_results:
        content_type = struct_data.get("content_type", "data")
        data = struct_data.get("data", {})
        
        if content_type == "table":
            table_text = data.get("text_representation", "")
            if table_text:
                structured_info.append(f"Table data: {table_text}")
        
        elif content_type == "chart":
            chart_text = data.get("text_representation", "")
            if chart_text:
                structured_info.append(f"Chart data: {chart_text}")
        
        elif content_type == "numeric":
            numeric_data = data
            value = numeric_data.get("value")
            original_text = numeric_data.get("original_text", "")
            if value and original_text:
                structured_info.append(f"Numeric data: {original_text}")
    
    # Combine all information
    # Prioritize structured numeric info for numeric queries
    is_numeric_query = any(kw in query.lower() for kw in ["how many", "how much", "number of", "count", "%", "percentage", "total"])
    all_info = (structured_info + key_sentences) if is_numeric_query else (key_sentences + structured_info)
    
    if all_info:
        summary = '. '.join(all_info[:6])
        if not summary.endswith(('.', '!', '?')):
            summary += '.'
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


@app.get("/api/v1/llm/status")
async def get_llm_status():
    """Check LLM availability and configuration status."""
    try:
        llm_status = {
            "openai_configured": False,
            "local_llm_configured": False,
            "local_llm_available": False,
            "response_mode": "extractive_summary",
            "llm_model": None,
            "base_url": None
        }
        
        # Check OpenAI configuration
        if settings.openai_api_key and settings.openai_api_key != "your_openai_api_key_here":
            llm_status["openai_configured"] = True
            llm_status["response_mode"] = "llm_generated"
            llm_status["llm_model"] = settings.default_llm_model
        
        # Check local LLM configuration
        if settings.use_local_llm and settings.local_llm_base_url:
            llm_status["local_llm_configured"] = True
            llm_status["llm_model"] = settings.local_llm_model or settings.default_llm_model
            llm_status["base_url"] = settings.local_llm_base_url
            
            # Test if local LLM is actually running
            import aiohttp
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        settings.local_llm_base_url + "/models", 
                        timeout=aiohttp.ClientTimeout(total=3)
                    ) as resp:
                        if resp.status == 200:
                            llm_status["local_llm_available"] = True
                            llm_status["response_mode"] = "llm_generated"
            except Exception:
                llm_status["local_llm_available"] = False
        
        # Determine overall status
        if llm_status["openai_configured"] or llm_status["local_llm_available"]:
            llm_status["status"] = "available"
        elif llm_status["local_llm_configured"]:
            llm_status["status"] = "configured_but_unavailable"
        else:
            llm_status["status"] = "not_configured"
            
        return llm_status
        
    except Exception as e:
        logger.error(f"Error checking LLM status: {e}")
        return {
            "status": "error",
            "error": str(e),
            "response_mode": "extractive_summary"
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
