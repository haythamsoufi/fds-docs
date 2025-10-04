"""Background tasks for document embedding and indexing."""

from typing import Optional
from celery import shared_task

from src.services.celery_app import celery_app
from src.core.config import settings
from src.core.database import get_db_session


@shared_task(name="embed_document_chunks")
def embed_document_chunks(document_id: Optional[str] = None) -> str:
    """Trigger embedding for a specific document or all documents."""
    import asyncio
    from src.services.embedding_service import EmbeddingService, VectorStore
    from src.core.models import ChunkModel
    from sqlalchemy import select

    async def _run():
        embedding_service = EmbeddingService()
        await embedding_service.initialize()
        vector_store = VectorStore(embedding_service)
        async with get_db_session() as session:
            if document_id:
                result = await session.execute(select(ChunkModel).where(ChunkModel.document_id == document_id))
            else:
                result = await session.execute(select(ChunkModel))
            chunks = result.scalars().all()
            if not chunks:
                return "no_chunks"
            await vector_store.add_documents(chunks)
            return f"embedded:{len(chunks)}"

    return asyncio.run(_run())


@shared_task(name="process_documents_directory")
def process_documents_directory() -> str:
    """Trigger processing of the documents directory."""
    import asyncio
    from src.services.document_processor import DocumentProcessor

    async def _run():
        processor = DocumentProcessor()
        return str(await processor.process_directory(settings.documents_path))

    return asyncio.run(_run())


