"""Service for storing and retrieving structured data (tables and charts)."""

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import sqlite3
from datetime import datetime

from src.core.config import settings
from src.core.database import get_db_session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

logger = logging.getLogger(__name__)


class StructuredDataService:
    """Service for managing structured data extracted from documents."""
    
    def __init__(self):
        self.structured_db_path = Path(settings.documents_path) / "structured_data.db"
        self._ensure_db_exists()
    
    def _ensure_db_exists(self):
        """Ensure the structured data database exists and is initialized."""
        try:
            self.structured_db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Initialize SQLite database for structured data
            with sqlite3.connect(str(self.structured_db_path)) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS structured_data (
                        id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        content_type TEXT NOT NULL,
                        page_number INTEGER,
                        data_json TEXT NOT NULL,
                        searchable_text TEXT,
                        metadata_json TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for faster queries
                conn.execute("CREATE INDEX IF NOT EXISTS idx_document_id ON structured_data(document_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_content_type ON structured_data(content_type)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_page_number ON structured_data(page_number)")
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error initializing structured data database: {e}")
    
    async def store_structured_data(
        self, 
        document_id: str, 
        structured_items: List[Dict[str, Any]], 
        content_type: str
    ) -> List[str]:
        """Store structured data (tables or charts) in the database."""
        stored_ids = []
        
        try:
            loop = asyncio.get_event_loop()
            
            def store_data():
                with sqlite3.connect(str(self.structured_db_path)) as conn:
                    for item in structured_items:
                        try:
                            item_id = item.get('table_id') or item.get('chart_id', f"{content_type}_{len(stored_ids)}")
                            
                            # Prepare data for storage
                            data_json = json.dumps(item, indent=2)
                            searchable_text = self._extract_searchable_text(item)
                            metadata_json = json.dumps({
                                "extraction_method": item.get("extraction_method"),
                                "rows": item.get("rows"),
                                "columns": item.get("columns"),
                                "page": item.get("page"),
                                "bbox": item.get("bbox"),
                                "confidence": item.get("confidence")
                            })
                            
                            # Insert or update record
                            conn.execute("""
                                INSERT OR REPLACE INTO structured_data 
                                (id, document_id, content_type, page_number, data_json, searchable_text, metadata_json, updated_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                item_id,
                                document_id,
                                content_type,
                                item.get("page"),
                                data_json,
                                searchable_text,
                                metadata_json,
                                datetime.utcnow().isoformat()
                            ))
                            
                            stored_ids.append(item_id)
                            
                        except Exception as e:
                            logger.warning(f"Error storing structured data item: {e}")
                            continue
                    
                    conn.commit()
                    
                return stored_ids
            
            stored_ids = await loop.run_in_executor(None, store_data)
            logger.info(f"Stored {len(stored_ids)} {content_type} items for document {document_id}")
            
        except Exception as e:
            logger.error(f"Error storing structured data: {e}")
        
        return stored_ids
    
    def _extract_searchable_text(self, item: Dict[str, Any]) -> str:
        """Extract searchable text from structured data item."""
        text_parts = []
        
        # Add text representation if available
        if item.get("text_representation"):
            text_parts.append(item["text_representation"])
        
        # Add headers for tables
        if item.get("headers"):
            text_parts.append(" ".join(item["headers"]))
        
        # Add numeric values for charts
        if item.get("numeric_values"):
            for num_data in item["numeric_values"]:
                text_parts.append(num_data.get("original_text", ""))
        
        # Add chart elements
        if item.get("chart_elements"):
            for element_type, elements in item["chart_elements"].items():
                text_parts.extend(elements)
        
        # Add raw text if available
        if item.get("raw_text"):
            text_parts.append(item["raw_text"])
        
        return " ".join(filter(None, text_parts))
    
    async def get_structured_data(
        self, 
        document_id: Optional[str] = None,
        content_type: Optional[str] = None,
        page_number: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve structured data with optional filters."""
        try:
            loop = asyncio.get_event_loop()
            
            def query_data():
                with sqlite3.connect(str(self.structured_db_path)) as conn:
                    conn.row_factory = sqlite3.Row
                    
                    # Build query
                    query = "SELECT * FROM structured_data WHERE 1=1"
                    params = []
                    
                    if document_id:
                        query += " AND document_id = ?"
                        params.append(document_id)
                    
                    if content_type:
                        query += " AND content_type = ?"
                        params.append(content_type)
                    
                    if page_number is not None:
                        query += " AND page_number = ?"
                        params.append(page_number)
                    
                    query += " ORDER BY page_number, id"
                    
                    cursor = conn.execute(query, params)
                    rows = cursor.fetchall()
                    
                    # Convert to dictionaries
                    results = []
                    for row in rows:
                        result = {
                            "id": row["id"],
                            "document_id": row["document_id"],
                            "content_type": row["content_type"],
                            "page_number": row["page_number"],
                            "data": json.loads(row["data_json"]),
                            "searchable_text": row["searchable_text"],
                            "metadata": json.loads(row["metadata_json"]) if row["metadata_json"] else {},
                            "created_at": row["created_at"],
                            "updated_at": row["updated_at"]
                        }
                        results.append(result)
                    
                    return results
            
            results = await loop.run_in_executor(None, query_data)
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving structured data: {e}")
            return []
    
    async def search_structured_data(
        self, 
        query: str, 
        content_types: Optional[List[str]] = None,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search structured data by text content."""
        try:
            loop = asyncio.get_event_loop()
            
            def search_data():
                with sqlite3.connect(str(self.structured_db_path)) as conn:
                    conn.row_factory = sqlite3.Row
                    
                    # Build search query
                    search_query = """
                        SELECT * FROM structured_data 
                        WHERE searchable_text LIKE ? 
                    """
                    params = [f"%{query}%"]
                    
                    if content_types:
                        placeholders = ",".join("?" * len(content_types))
                        search_query += f" AND content_type IN ({placeholders})"
                        params.extend(content_types)
                    
                    if document_ids:
                        placeholders = ",".join("?" * len(document_ids))
                        search_query += f" AND document_id IN ({placeholders})"
                        params.extend(document_ids)
                    
                    search_query += " ORDER BY page_number, id"
                    
                    cursor = conn.execute(search_query, params)
                    rows = cursor.fetchall()
                    
                    # Convert to dictionaries
                    results = []
                    for row in rows:
                        result = {
                            "id": row["id"],
                            "document_id": row["document_id"],
                            "content_type": row["content_type"],
                            "page_number": row["page_number"],
                            "data": json.loads(row["data_json"]),
                            "searchable_text": row["searchable_text"],
                            "metadata": json.loads(row["metadata_json"]) if row["metadata_json"] else {},
                            "created_at": row["created_at"],
                            "updated_at": row["updated_at"]
                        }
                        results.append(result)
                    
                    return results
            
            results = await loop.run_in_executor(None, search_data)
            return results
            
        except Exception as e:
            logger.error(f"Error searching structured data: {e}")
            return []
    
    async def get_numeric_data(
        self, 
        document_id: Optional[str] = None,
        content_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get all numeric data from structured sources."""
        try:
            # Get all structured data
            structured_data = await self.get_structured_data(
                document_id=document_id,
                content_type=content_types[0] if content_types and len(content_types) == 1 else None
            )
            
            numeric_data = []
            
            for item in structured_data:
                data = item["data"]
                
                # Extract numeric values from tables
                if item["content_type"] == "table" and data.get("data"):
                    table_data = data["data"]
                    for row_idx, row in enumerate(table_data):
                        for col_idx, cell in enumerate(row):
                            if self._is_numeric(cell):
                                numeric_data.append({
                                    "value": float(cell.replace(",", "")),
                                    "original_text": cell,
                                    "type": "table_cell",
                                    "table_id": item["id"],
                                    "document_id": item["document_id"],
                                    "page_number": item["page_number"],
                                    "row": row_idx,
                                    "column": col_idx,
                                    "context": {
                                        "headers": data.get("headers", []),
                                        "row_data": row
                                    }
                                })
                
                # Extract numeric values from charts
                elif item["content_type"] == "chart" and data.get("numeric_values"):
                    for num_data in data["numeric_values"]:
                        numeric_data.append({
                            "value": num_data["value"],
                            "original_text": num_data["original_text"],
                            "type": f"chart_{num_data['type']}",
                            "chart_id": item["id"],
                            "document_id": item["document_id"],
                            "page_number": item["page_number"],
                            "context": {
                                "chart_elements": data.get("chart_elements", {}),
                                "raw_text": data.get("raw_text", "")
                            }
                        })
            
            return numeric_data
            
        except Exception as e:
            logger.error(f"Error extracting numeric data: {e}")
            return []
    
    def _is_numeric(self, text: str) -> bool:
        """Check if text represents a numeric value."""
        if not text or not text.strip():
            return False
        
        text = text.strip().replace(",", "").replace("$", "").replace("%", "")
        
        try:
            float(text)
            return True
        except ValueError:
            return False
    
    async def delete_document_data(self, document_id: str):
        """Delete all structured data for a document."""
        try:
            loop = asyncio.get_event_loop()
            
            def delete_data():
                with sqlite3.connect(str(self.structured_db_path)) as conn:
                    cursor = conn.execute(
                        "DELETE FROM structured_data WHERE document_id = ?",
                        (document_id,)
                    )
                    deleted_count = cursor.rowcount
                    conn.commit()
                    return deleted_count
            
            deleted_count = await loop.run_in_executor(None, delete_data)
            logger.info(f"Deleted {deleted_count} structured data items for document {document_id}")
            
        except Exception as e:
            logger.error(f"Error deleting structured data for document {document_id}: {e}")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored structured data."""
        try:
            loop = asyncio.get_event_loop()
            
            def get_stats():
                with sqlite3.connect(str(self.structured_db_path)) as conn:
                    conn.row_factory = sqlite3.Row
                    
                    # Get total counts
                    total_cursor = conn.execute("SELECT COUNT(*) as count FROM structured_data")
                    total_count = total_cursor.fetchone()["count"]
                    
                    # Get counts by type
                    type_cursor = conn.execute("""
                        SELECT content_type, COUNT(*) as count 
                        FROM structured_data 
                        GROUP BY content_type
                    """)
                    type_counts = {row["content_type"]: row["count"] for row in type_cursor.fetchall()}
                    
                    # Get document counts
                    doc_cursor = conn.execute("""
                        SELECT COUNT(DISTINCT document_id) as count 
                        FROM structured_data
                    """)
                    document_count = doc_cursor.fetchone()["count"]
                    
                    return {
                        "total_items": total_count,
                        "by_content_type": type_counts,
                        "documents_with_structured_data": document_count
                    }
            
            stats = await loop.run_in_executor(None, get_stats)
            return stats
            
        except Exception as e:
            logger.error(f"Error getting structured data statistics: {e}")
            return {}


# Global instance
structured_data_service = StructuredDataService()
