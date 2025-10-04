"""Multimodal document processor that handles text, tables, and charts."""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json

from .table_extraction_service import table_extraction_service
from .chart_extraction_service import chart_extraction_service
from .text_splitter import TextChunk
from .document_processor import DocumentProcessor

logger = logging.getLogger(__name__)


class MultimodalDocumentProcessor:
    """Enhanced document processor that extracts and processes text, tables, and charts."""
    
    def __init__(self):
        self.base_processor = DocumentProcessor()
        self.table_service = table_extraction_service
        self.chart_service = chart_extraction_service
    
    async def process_document_multimodal(self, filepath: str) -> Optional[str]:
        """Process a document extracting text, tables, and charts."""
        file_extension = Path(filepath).suffix.lower()
        
        if file_extension != '.pdf':
            # For non-PDF files, use standard processing
            return await self.base_processor.process_document(filepath)
        
        try:
            # Extract all content types in parallel
            text_task = self._extract_text_content(filepath)
            tables_task = self._extract_tables(filepath)
            charts_task = self._extract_charts(filepath)
            
            # Wait for all extractions to complete
            text_content, text_metadata = await text_task
            tables, table_metadata = await tables_task
            charts, chart_metadata = await charts_task
            
            # Combine all content
            combined_content = await self._combine_multimodal_content(
                text_content, tables, charts
            )
            
            # Create enhanced metadata
            enhanced_metadata = {
                **text_metadata,
                "multimodal_processing": True,
                "tables_extracted": len(tables),
                "charts_extracted": len(charts),
                "table_metadata": table_metadata,
                "chart_metadata": chart_metadata,
                "content_types": self._get_content_types(text_content, tables, charts)
            }
            
            # Store the document using the base processor
            return await self._store_multimodal_document(
                filepath, combined_content, enhanced_metadata
            )
            
        except Exception as e:
            logger.error(f"Error in multimodal document processing: {e}")
            # Fallback to standard processing
            return await self.base_processor.process_document(filepath)
    
    async def _extract_text_content(self, filepath: str) -> Tuple[str, Dict[str, Any]]:
        """Extract text content using the base processor."""
        try:
            return await self.base_processor.extract_text_content(filepath)
        except Exception as e:
            logger.warning(f"Text extraction failed: {e}")
            return "", {"text_extraction_error": str(e)}
    
    async def _extract_tables(self, filepath: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Extract tables from the document."""
        try:
            return await self.table_service.extract_tables_from_pdf(filepath)
        except Exception as e:
            logger.warning(f"Table extraction failed: {e}")
            return [], {"table_extraction_error": str(e)}
    
    async def _extract_charts(self, filepath: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Extract charts from the document."""
        try:
            return await self.chart_service.extract_charts_from_pdf(filepath)
        except Exception as e:
            logger.warning(f"Chart extraction failed: {e}")
            return [], {"chart_extraction_error": str(e)}
    
    async def _combine_multimodal_content(self, text_content: str, tables: List[Dict[str, Any]], charts: List[Dict[str, Any]]) -> str:
        """Combine text, table, and chart content into a unified format."""
        content_parts = []
        
        # Add main text content
        if text_content.strip():
            content_parts.append("=== DOCUMENT TEXT ===")
            content_parts.append(text_content.strip())
            content_parts.append("")
        
        # Add tables
        if tables:
            content_parts.append("=== TABLES ===")
            for i, table in enumerate(tables, 1):
                content_parts.append(f"--- Table {i} (Page {table.get('page', 'Unknown')}) ---")
                content_parts.append(table.get('text_representation', ''))
                
                # Add structured data reference
                content_parts.append(f"[Structured table data available - ID: {table.get('table_id', 'unknown')}]")
                content_parts.append("")
        
        # Add charts
        if charts:
            content_parts.append("=== CHARTS AND FIGURES ===")
            for i, chart in enumerate(charts, 1):
                content_parts.append(f"--- Chart {i} (Page {chart.get('page', 'Unknown')}) ---")
                content_parts.append(chart.get('text_representation', ''))
                
                # Add structured data reference
                content_parts.append(f"[Structured chart data available - ID: {chart.get('chart_id', 'unknown')}]")
                content_parts.append("")
        
        return "\n".join(content_parts)
    
    def _get_content_types(self, text_content: str, tables: List[Dict[str, Any]], charts: List[Dict[str, Any]]) -> List[str]:
        """Determine what types of content were found in the document."""
        content_types = []
        
        if text_content.strip():
            content_types.append("text")
        
        if tables:
            content_types.append("tables")
        
        if charts:
            content_types.append("charts")
        
        return content_types
    
    async def _store_multimodal_document(self, filepath: str, combined_content: str, metadata: Dict[str, Any]) -> Optional[str]:
        """Store the multimodal document using the base processor's logic."""
        try:
            # Use the base processor's document creation/update logic
            from src.core.database import get_db_session
            from src.core.models import DocumentModel
            from sqlalchemy import select
            import time
            
            # Compute file hash
            file_hash = await self.base_processor.compute_file_hash(filepath)
            file_stats = Path(filepath).stat()
            
            async with get_db_session() as session:
                # Check if document exists
                existing_doc = await self._get_document_by_path(session, filepath)
                
                if existing_doc:
                    # Update existing document
                    document_id = await self._update_document(
                        session, existing_doc, file_hash, combined_content, metadata
                    )
                else:
                    # Create new document
                    document_id = await self._create_document(
                        session, filepath, file_hash, file_stats.st_size, combined_content, metadata
                    )
                
                return str(document_id)
                
        except Exception as e:
            logger.error(f"Error storing multimodal document: {e}")
            raise
    
    async def _get_document_by_path(self, session, filepath: str):
        """Get existing document by file path."""
        from src.core.models import DocumentModel
        from sqlalchemy import select
        
        result = await session.execute(
            select(DocumentModel).where(DocumentModel.filepath == filepath)
        )
        return result.scalar_one_or_none()
    
    async def _update_document(self, session, existing_doc, file_hash: str, content: str, metadata: Dict[str, Any]):
        """Update existing document."""
        from datetime import datetime
        
        existing_doc.file_hash = file_hash
        existing_doc.content = content
        existing_doc.extra_metadata = metadata
        existing_doc.updated_at = datetime.utcnow()
        existing_doc.status = "processing"
        
        # Delete old chunks
        from src.core.models import ChunkModel
        from sqlalchemy import delete
        
        await session.execute(
            delete(ChunkModel).where(ChunkModel.document_id == existing_doc.id)
        )
        
        # Create new chunks
        await self._create_chunks(session, existing_doc.id, content, metadata)
        
        await session.commit()
        return existing_doc.id
    
    async def _create_document(self, session, filepath: str, file_hash: str, file_size: int, content: str, metadata: Dict[str, Any]):
        """Create new document."""
        from datetime import datetime
        from src.core.models import DocumentModel, DocumentCreate
        
        doc_data = DocumentCreate(
            filename=Path(filepath).name,
            filepath=filepath,
            file_size=file_size,
            mime_type="application/pdf",
            content=content,
            extra_metadata=metadata
        )
        
        document = DocumentModel(**doc_data.dict())
        document.file_hash = file_hash
        document.status = "processing"
        document.created_at = datetime.utcnow()
        document.updated_at = datetime.utcnow()
        
        session.add(document)
        await session.flush()  # Get the ID
        
        # Create chunks
        await self._create_chunks(session, document.id, content, metadata)
        
        await session.commit()
        return document.id
    
    async def _create_chunks(self, session, document_id: str, content: str, metadata: Dict[str, Any]):
        """Create chunks from the multimodal content."""
        from src.core.models import ChunkModel, ChunkCreate
        from datetime import datetime
        
        # Split content into chunks
        chunks = await self._split_multimodal_content(content, metadata)
        
        for i, chunk_data in enumerate(chunks):
            chunk_metadata = {
                **metadata,
                "chunk_index": i,
                "content_type": chunk_data.get("content_type", "text"),
                "structured_data_id": chunk_data.get("structured_data_id"),
                "source_element": chunk_data.get("source_element")
            }
            
            chunk = ChunkModel(
                document_id=document_id,
                chunk_index=i,
                content=chunk_data["content"],
                extra_metadata=chunk_metadata,
                created_at=datetime.utcnow()
            )
            
            session.add(chunk)
    
    async def _split_multimodal_content(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split multimodal content into chunks, preserving structured data references."""
        from .text_splitter import AdaptiveTextSplitter
        
        splitter = AdaptiveTextSplitter()
        
        # Split the combined content
        text_chunks = await splitter.split_text(content, metadata)
        
        chunks = []
        for chunk in text_chunks:
            chunk_dict = {
                "content": chunk.content,
                "content_type": "multimodal",
                "metadata": chunk.metadata
            }
            
            # Check if this chunk references structured data
            if "[Structured table data available" in chunk.content:
                chunk_dict["content_type"] = "table_reference"
                # Extract table ID from the reference
                import re
                match = re.search(r'ID: ([^)]+)', chunk.content)
                if match:
                    chunk_dict["structured_data_id"] = match.group(1)
                    chunk_dict["source_element"] = "table"
            
            elif "[Structured chart data available" in chunk.content:
                chunk_dict["content_type"] = "chart_reference"
                # Extract chart ID from the reference
                import re
                match = re.search(r'ID: ([^)]+)', chunk.content)
                if match:
                    chunk_dict["structured_data_id"] = match.group(1)
                    chunk_dict["source_element"] = "chart"
            
            chunks.append(chunk_dict)
        
        return chunks


# Global instance
multimodal_document_processor = MultimodalDocumentProcessor()
