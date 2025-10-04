"""Document management routes."""

import asyncio
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from src.core.database import get_db
from src.core.models import DocumentModel, ChunkModel, DocumentResponse, ProcessingStatus

router = APIRouter()


@router.get("/", response_model=List[DocumentResponse])
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db)
):
    """List documents with pagination and filtering."""
    query = select(DocumentModel)
    
    if status:
        query = query.where(DocumentModel.status == status)
        
    query = query.offset(skip).limit(limit).order_by(DocumentModel.created_at.desc())
    
    result = await db.execute(query)
    documents = result.scalars().all()
    
    return [DocumentResponse.model_validate(doc, from_attributes=True) for doc in documents]


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific document."""
    result = await db.execute(
        select(DocumentModel).where(DocumentModel.id == document_id)
    )
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
        
    return DocumentResponse.model_validate(document, from_attributes=True)


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Delete a document and its chunks."""
    import os
    from pathlib import Path
    from sqlalchemy import delete
    from src.core.config import settings
    from src.services.embedding_service import EmbeddingService, VectorStore
    
    result = await db.execute(
        select(DocumentModel).where(DocumentModel.id == document_id)
    )
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        # 1. Delete physical file if it exists
        if document.filepath and os.path.exists(document.filepath):
            try:
                os.remove(document.filepath)
            except OSError as e:
                print(f"Warning: Could not delete file {document.filepath}: {e}")
        
        # 2. Delete chunks from database
        await db.execute(
            delete(ChunkModel).where(ChunkModel.document_id == document_id)
        )
        
        # 3. Delete chunks from vector database (with timeout and fallback)
        deleted_vector_count = 0
        try:
            embedding_service = EmbeddingService()
            await embedding_service.initialize()
            vector_store = VectorStore(embedding_service)
            
            # Use a shorter timeout for delete operations to prevent hanging
            try:
                deleted_vector_count = await asyncio.wait_for(
                    vector_store.delete_document(document_id),
                    timeout=15.0  # 15 second timeout for vector deletion
                )
                print(f"Deleted {deleted_vector_count} chunks from vector database")
            except asyncio.TimeoutError:
                print(f"Warning: Vector deletion timed out for document {document_id}")
                # Don't fail the entire operation if vector deletion times out
                deleted_vector_count = -1  # Indicate timeout occurred
                
        except Exception as e:
            print(f"Warning: Could not delete chunks from vector database: {e}")
            deleted_vector_count = -1  # Indicate error occurred
        
        # 4. Delete document record from database
        await db.delete(document)
        await db.commit()
        
        return {
            "message": "Document deleted successfully",
            "deleted_file": bool(document.filepath and os.path.exists(document.filepath)),
            "vector_chunks_deleted": deleted_vector_count
        }
        
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@router.get("/status/summary", response_model=ProcessingStatus)
async def get_processing_status(db: AsyncSession = Depends(get_db)):
    """Get document processing status summary."""
    # Count documents by status
    total_result = await db.execute(select(func.count(DocumentModel.id)))
    total_documents = total_result.scalar()
    
    processed_result = await db.execute(
        select(func.count(DocumentModel.id)).where(DocumentModel.status == "completed")
    )
    processed_documents = processed_result.scalar()
    
    processing_result = await db.execute(
        select(func.count(DocumentModel.id)).where(DocumentModel.status == "processing")
    )
    processing_documents = processing_result.scalar()
    
    failed_result = await db.execute(
        select(func.count(DocumentModel.id)).where(DocumentModel.status == "failed")
    )
    failed_documents = failed_result.scalar()
    
    return ProcessingStatus(
        total_documents=total_documents or 0,
        processed_documents=processed_documents or 0,
        failed_documents=failed_documents or 0,
        processing_documents=processing_documents or 0,
        last_updated=datetime.utcnow().isoformat()
    )


@router.post("/cleanup")
async def cleanup_orphaned_files_and_chunks(db: AsyncSession = Depends(get_db)):
    """Clean up orphaned files and chunks that are no longer referenced."""
    import os
    import asyncio
    from pathlib import Path
    from sqlalchemy import delete
    from src.core.config import settings
    from src.services.embedding_service import EmbeddingService, VectorStore
    
    cleanup_stats = {
        "orphaned_files_deleted": 0,
        "orphaned_chunks_deleted": 0,
        "vector_chunks_cleaned": 0,
        "errors": []
    }
    
    try:
        # 1. Find and delete orphaned files
        documents_dir = Path(settings.documents_path)
        if documents_dir.exists():
            # Get all document filepaths from database
            result = await db.execute(select(DocumentModel.filepath))
            valid_filepaths = {row[0] for row in result.fetchall()}
            
            # Check for orphaned files
            for file_path in documents_dir.iterdir():
                if file_path.is_file():
                    if str(file_path) not in valid_filepaths:
                        try:
                            file_path.unlink()
                            cleanup_stats["orphaned_files_deleted"] += 1
                        except OSError as e:
                            cleanup_stats["errors"].append(f"Could not delete {file_path}: {e}")
        
        # 2. Find and delete orphaned chunks (chunks without valid document_id)
        result = await db.execute(select(DocumentModel.id))
        valid_document_ids = {row[0] for row in result.fetchall()}
        
        # Find chunks with invalid document_ids
        result = await db.execute(select(ChunkModel.document_id).distinct())
        chunk_document_ids = {row[0] for row in result.fetchall()}
        
        orphaned_document_ids = chunk_document_ids - valid_document_ids
        
        if orphaned_document_ids:
            # Delete orphaned chunks
            await db.execute(
                delete(ChunkModel).where(ChunkModel.document_id.in_(orphaned_document_ids))
            )
            cleanup_stats["orphaned_chunks_deleted"] = len(orphaned_document_ids)
            
            # Also clean up vector database (with timeout protection)
            try:
                embedding_service = EmbeddingService()
                await embedding_service.initialize()
                vector_store = VectorStore(embedding_service)
                
                for doc_id in orphaned_document_ids:
                    try:
                        # Use timeout for each vector deletion to prevent hanging
                        deleted_count = await asyncio.wait_for(
                            vector_store.delete_document(doc_id),
                            timeout=10.0  # 10 second timeout per document
                        )
                        cleanup_stats["vector_chunks_cleaned"] += max(0, deleted_count)
                    except asyncio.TimeoutError:
                        cleanup_stats["errors"].append(f"Vector cleanup timeout for document {doc_id}")
                    except Exception as e:
                        cleanup_stats["errors"].append(f"Vector cleanup error for document {doc_id}: {e}")
                        
            except Exception as e:
                cleanup_stats["errors"].append(f"Vector cleanup initialization error: {e}")
        
        await db.commit()
        
        return {
            "message": "Cleanup completed",
            "stats": cleanup_stats
        }
        
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


@router.get("/status/summary", response_model=ProcessingStatus)
async def get_processing_status(db: AsyncSession = Depends(get_db)):
    """Get document processing status summary."""
    from datetime import datetime
    
    # Count documents by status
    result = await db.execute(
        select(
            DocumentModel.status,
            func.count(DocumentModel.id).label('count')
        ).group_by(DocumentModel.status)
    )
    
    status_counts = {row.status: row.count for row in result}
    
    return ProcessingStatus(
        total_documents=sum(status_counts.values()),
        processed_documents=status_counts.get('completed', 0),
        failed_documents=status_counts.get('failed', 0),
        processing_documents=status_counts.get('processing', 0),
        last_updated=datetime.utcnow()
    )
