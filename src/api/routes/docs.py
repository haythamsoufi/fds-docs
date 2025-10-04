"""Documentation API routes for serving markdown files."""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/docs", tags=["documentation"])

# Documentation directory
DOCS_DIR = Path(__file__).parent.parent.parent.parent / "docs"


class DocumentationFile(BaseModel):
    filename: str
    title: str
    description: str
    category: str
    size: Optional[int] = None
    last_modified: Optional[str] = None


class DocumentationList(BaseModel):
    files: List[DocumentationFile]


@router.get("/", response_model=DocumentationList)
async def list_documentation():
    """List all available documentation files."""
    try:
        if not DOCS_DIR.exists():
            raise HTTPException(status_code=404, detail="Documentation directory not found")
        
        files = []
        
        # Define documentation metadata
        doc_metadata = {
            "MULTIMODAL_PROCESSING_GUIDE.md": {
                "title": "Multimodal Processing Guide",
                "description": "Complete guide to table and chart extraction from PDFs",
                "category": "Processing"
            },
            "MIGRATION_GUIDE.md": {
                "title": "Migration Guide",
                "description": "How to migrate from previous versions",
                "category": "Setup"
            },
            "OCR_REPLACEMENT_GUIDE.md": {
                "title": "OCR Replacement Guide",
                "description": "Upgrading OCR capabilities and configuration",
                "category": "Configuration"
            },
            "RAG_UPGRADE_PLAN.md": {
                "title": "RAG Upgrade Plan",
                "description": "Retrieval-Augmented Generation system improvements",
                "category": "Architecture"
            },
            "ROLLBACK_PROCEDURES.md": {
                "title": "Rollback Procedures",
                "description": "How to rollback changes if needed",
                "category": "Operations"
            }
        }
        
        # Scan for markdown files
        for file_path in DOCS_DIR.glob("*.md"):
            if file_path.is_file():
                filename = file_path.name
                metadata = doc_metadata.get(filename, {
                    "title": filename.replace(".md", "").replace("_", " ").title(),
                    "description": f"Documentation file: {filename}",
                    "category": "General"
                })
                
                stat = file_path.stat()
                files.append(DocumentationFile(
                    filename=filename,
                    title=metadata["title"],
                    description=metadata["description"],
                    category=metadata["category"],
                    size=stat.st_size,
                    last_modified=stat.st_mtime
                ))
        
        # Sort by category, then by title
        files.sort(key=lambda x: (x.category, x.title))
        
        return DocumentationList(files=files)
        
    except Exception as e:
        logger.error(f"Error listing documentation: {e}")
        raise HTTPException(status_code=500, detail="Failed to list documentation files")


@router.get("/health")
async def docs_health():
    """Check if the documentation service is healthy."""
    try:
        if not DOCS_DIR.exists():
            return {
                "status": "unhealthy",
                "message": "Documentation directory not found",
                "docs_directory": str(DOCS_DIR)
            }
        
        # Check if we can read the directory
        files = list(DOCS_DIR.glob("*.md"))
        
        return {
            "status": "healthy",
            "message": f"Documentation service is working",
            "docs_directory": str(DOCS_DIR),
            "files_count": len(files),
            "available_files": [f.name for f in files]
        }
        
    except Exception as e:
        logger.error(f"Documentation health check failed: {e}")
        return {
            "status": "unhealthy",
            "message": f"Documentation service error: {str(e)}",
            "docs_directory": str(DOCS_DIR)
        }


@router.get("/{filename}")
async def get_documentation(filename: str):
    """Get a specific documentation file."""
    try:
        # Security check - only allow markdown files
        if not filename.endswith('.md'):
            raise HTTPException(status_code=400, detail="Only markdown files are supported")
        
        # Prevent directory traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        file_path = DOCS_DIR / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Documentation file not found: {filename}")
        
        if not file_path.is_file():
            raise HTTPException(status_code=400, detail="Invalid file path")
        
        # Return the markdown file as plain text
        return FileResponse(
            path=str(file_path),
            media_type="text/plain",
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving documentation file {filename}: {e}")
        raise HTTPException(status_code=500, detail="Failed to serve documentation file")


@router.get("/{filename}/content")
async def get_documentation_content(filename: str):
    """Get the content of a documentation file as plain text."""
    try:
        # Security check - only allow markdown files
        if not filename.endswith('.md'):
            raise HTTPException(status_code=400, detail="Only markdown files are supported")
        
        # Prevent directory traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        file_path = DOCS_DIR / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Documentation file not found: {filename}")
        
        if not file_path.is_file():
            raise HTTPException(status_code=400, detail="Invalid file path")
        
        # Read and return the content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return PlainTextResponse(content)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading documentation file {filename}: {e}")
        raise HTTPException(status_code=500, detail="Failed to read documentation file")


@router.get("/categories/list")
async def list_categories():
    """List all documentation categories."""
    categories = [
        {
            "id": "processing",
            "name": "Document Processing",
            "description": "How documents are processed and indexed",
            "count": 1
        },
        {
            "id": "setup",
            "name": "Setup & Configuration",
            "description": "Installation and configuration guides",
            "count": 1
        },
        {
            "id": "configuration",
            "name": "Configuration",
            "description": "System configuration and customization",
            "count": 1
        },
        {
            "id": "architecture",
            "name": "System Architecture",
            "description": "Technical architecture and components",
            "count": 1
        },
        {
            "id": "operations",
            "name": "Operations",
            "description": "Operational procedures and maintenance",
            "count": 1
        }
    ]
    
    return {"categories": categories}
