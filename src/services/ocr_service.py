"""OCR service for scanned PDFs and table extraction."""

import logging
import io
import tempfile
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING, Union
from pathlib import Path
import asyncio

from src.core.config import settings

logger = logging.getLogger(__name__)

# Type hints for when PIL is available
if TYPE_CHECKING:
    from PIL import Image

# Optional imports with fallbacks
try:
    import easyocr
    from PIL import Image
    import pdf2image
    import pdfplumber
    import requests
    import json
    from io import BytesIO
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("OCR dependencies not available. Install easyocr, Pillow, pdf2image, pdfplumber for OCR support.")


class OCRService:
    """Service for OCR and table extraction from scanned PDFs using EasyOCR and cloud alternatives."""
    
    def __init__(self):
        self.ocr_available = OCR_AVAILABLE
        self.ocr_engine = None
        self.cloud_ocr_available = False
        if self.ocr_available:
            self._initialize_ocr()
    
    def _initialize_ocr(self):
        """Initialize OCR engines (EasyOCR primary, cloud fallback)."""
        try:
            # Check if OCR is explicitly disabled
            if not settings.ocr_enabled:
                logger.info("OCR disabled by configuration (OCR_ENABLED=false)")
                self.ocr_available = False
                return
            
            # Initialize EasyOCR (Docker-friendly, no system dependencies)
            self.ocr_engine = easyocr.Reader(['en'], gpu=False)  # CPU-only for Docker compatibility
            logger.info("EasyOCR initialized successfully")
            
            # Check for cloud OCR credentials
            self._check_cloud_ocr_credentials()
            
        except Exception as e:
            logger.warning(f"EasyOCR initialization failed: {e}")
            logger.info("Falling back to cloud OCR or disabling OCR")
            self.ocr_available = False
            self._check_cloud_ocr_credentials()
    
    def _check_cloud_ocr_credentials(self):
        """Check if cloud OCR credentials are available."""
        # Check for Azure Computer Vision credentials
        azure_key = getattr(settings, 'azure_vision_key', None)
        azure_endpoint = getattr(settings, 'azure_vision_endpoint', None)
        
        if azure_key and azure_endpoint:
            self.cloud_ocr_available = True
            self.azure_key = azure_key
            self.azure_endpoint = azure_endpoint
            logger.info("Azure Computer Vision OCR credentials found")
            return
        
        # Check for Google Vision API credentials
        google_credentials = getattr(settings, 'google_vision_credentials', None)
        if google_credentials:
            self.cloud_ocr_available = True
            self.google_credentials = google_credentials
            logger.info("Google Vision API credentials found")
            return
        
        logger.info("No cloud OCR credentials found. OCR will be disabled if EasyOCR fails.")
    
    async def extract_text_with_ocr(self, filepath: str) -> Tuple[str, Dict[str, Any]]:
        """Extract text from scanned PDF using OCR."""
        if not self.ocr_available:
            logger.warning("OCR not available, returning empty text")
            return "", {"ocr_available": False, "error": "OCR dependencies not installed"}
        
        try:
            file_path = Path(filepath)
            if file_path.suffix.lower() != '.pdf':
                return "", {"error": "OCR currently only supports PDF files"}
            
            # Convert PDF to images
            images = await self._pdf_to_images(filepath)
            if not images:
                return "", {"error": "Could not convert PDF to images"}
            
            # Extract text from each image
            all_text = []
            page_metadata = []
            
            for i, image in enumerate(images):
                page_text = await self._extract_text_from_image(image)
                all_text.append(f"[Page {i+1}]\n{page_text}")
                
                page_metadata.append({
                    "page": i+1,
                    "text_length": len(page_text),
                    "has_text": len(page_text.strip()) > 0
                })
            
            combined_text = "\n\n".join(all_text)
            
            metadata = {
                "ocr_available": True,
                "extraction_method": "ocr",
                "page_count": len(images),
                "total_text_length": len(combined_text),
                "page_metadata": page_metadata,
                "has_extracted_text": len(combined_text.strip()) > 0
            }
            
            logger.info(f"OCR extraction completed: {len(combined_text)} characters from {len(images)} pages")
            return combined_text, metadata
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return "", {"ocr_available": True, "error": str(e)}
    
    async def extract_tables_from_pdf(self, filepath: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Extract tables from PDF using pdfplumber."""
        if not OCR_AVAILABLE:
            return [], {"error": "pdfplumber not available"}
        
        try:
            tables = []
            metadata = {
                "extraction_method": "pdfplumber",
                "total_tables": 0,
                "page_metadata": []
            }
            
            with pdfplumber.open(filepath) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    page_metadata = {
                        "page": page_num + 1,
                        "table_count": len(page_tables) if page_tables else 0
                    }
                    metadata["page_metadata"].append(page_metadata)
                    
                    if page_tables:
                        for table_num, table in enumerate(page_tables):
                            formatted_table = self._format_table(table, page_num + 1, table_num + 1)
                            tables.append(formatted_table)
            
            metadata["total_tables"] = len(tables)
            logger.info(f"Extracted {len(tables)} tables from PDF")
            return tables, metadata
            
        except Exception as e:
            logger.error(f"Table extraction failed: {e}")
            return [], {"error": str(e)}
    
    async def _pdf_to_images(self, filepath: str) -> List[Any]:
        """Convert PDF pages to images for OCR."""
        try:
            # Convert PDF to images with configurable DPI
            images = pdf2image.convert_from_path(
                filepath,
                dpi=settings.ocr_dpi,
                first_page=None,
                last_page=None,
                fmt='RGB'
            )
            return images
        except Exception as e:
            error_msg = str(e)
            if "invalid float value" in error_msg or "gray non-stroke color" in error_msg:
                logger.warning(f"PDF graphics-related error during image conversion: {error_msg}")
                logger.info("Attempting alternative PDF processing parameters")
                try:
                    # Try with different parameters for problematic PDFs
                    images = pdf2image.convert_from_path(
                        filepath,
                        dpi=150,  # Lower DPI
                        first_page=None,
                        last_page=None,
                        fmt='RGB',
                        use_pdftocairo=False,  # Disable Cairo backend
                        strict=False  # More lenient parsing
                    )
                    logger.info("Alternative PDF processing succeeded")
                    return images
                except Exception as fallback_error:
                    logger.error(f"Alternative PDF processing also failed: {fallback_error}")
                    return []
            else:
                logger.error(f"PDF to image conversion failed: {e}")
                return []
    
    async def _extract_text_from_image(self, image: Any) -> str:
        """Extract text from a single image using EasyOCR or cloud OCR."""
        try:
            # Try EasyOCR first
            if self.ocr_engine:
                loop = asyncio.get_event_loop()
                text = await loop.run_in_executor(
                    None,
                    self._extract_with_easyocr,
                    image
                )
                if text.strip():
                    return text.strip()
            
            # Fallback to cloud OCR
            if self.cloud_ocr_available:
                return await self._extract_with_cloud_ocr(image)
            
            return ""
            
        except Exception as e:
            logger.error(f"OCR text extraction failed: {e}")
            return ""
    
    def _extract_with_easyocr(self, image: Any) -> str:
        """Extract text using EasyOCR."""
        try:
            # Convert PIL image to numpy array if needed
            import numpy as np
            if hasattr(image, 'convert'):  # PIL Image
                image_array = np.array(image.convert('RGB'))
            else:
                image_array = np.array(image)
            
            # Run OCR
            results = self.ocr_engine.readtext(image_array)
            
            # Extract text from results
            text_parts = []
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Filter low-confidence results
                    text_parts.append(text)
            
            return ' '.join(text_parts)
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return ""
    
    async def _extract_with_cloud_ocr(self, image: Any) -> str:
        """Extract text using cloud OCR (Azure or Google)."""
        try:
            # Try Azure Computer Vision first
            if hasattr(self, 'azure_key') and hasattr(self, 'azure_endpoint'):
                return await self._extract_with_azure_vision(image)
            
            # Try Google Vision API
            if hasattr(self, 'google_credentials'):
                return await self._extract_with_google_vision(image)
            
            return ""
        except Exception as e:
            logger.error(f"Cloud OCR extraction failed: {e}")
            return ""
    
    async def _extract_with_azure_vision(self, image: Any) -> str:
        """Extract text using Azure Computer Vision."""
        try:
            # Convert image to bytes
            img_buffer = BytesIO()
            if hasattr(image, 'save'):  # PIL Image
                image.save(img_buffer, format='PNG')
            else:
                # Assume it's already bytes
                img_buffer.write(image)
            
            img_bytes = img_buffer.getvalue()
            
            # Call Azure Computer Vision API
            headers = {
                'Ocp-Apim-Subscription-Key': self.azure_key,
                'Content-Type': 'application/octet-stream'
            }
            
            params = {
                'language': 'en',
                'detectOrientation': 'true'
            }
            
            response = requests.post(
                f"{self.azure_endpoint}/vision/v3.2/read/analyze",
                headers=headers,
                params=params,
                data=img_bytes
            )
            
            if response.status_code == 202:  # Accepted
                # Get the operation location
                operation_url = response.headers['Operation-Location']
                
                # Poll for results
                import time
                for _ in range(10):  # Max 10 seconds
                    result_response = requests.get(
                        operation_url,
                        headers={'Ocp-Apim-Subscription-Key': self.azure_key}
                    )
                    
                    if result_response.status_code == 200:
                        result = result_response.json()
                        if result.get('status') == 'succeeded':
                            # Extract text from results
                            text_parts = []
                            for page in result.get('analyzeResult', {}).get('readResults', []):
                                for line in page.get('lines', []):
                                    text_parts.append(line.get('text', ''))
                            return ' '.join(text_parts)
                    
                    time.sleep(1)
            
            return ""
        except Exception as e:
            logger.error(f"Azure Vision OCR failed: {e}")
            return ""
    
    async def _extract_with_google_vision(self, image: Any) -> str:
        """Extract text using Google Vision API."""
        try:
            # This would require google-cloud-vision library
            # For now, return empty string and log the limitation
            logger.warning("Google Vision API integration not implemented yet")
            return ""
        except Exception as e:
            logger.error(f"Google Vision OCR failed: {e}")
            return ""
    
    def _format_table(self, table_data: List[List[str]], page_num: int, table_num: int) -> Dict[str, Any]:
        """Format extracted table data."""
        if not table_data:
            return {
                "page": page_num,
                "table_number": table_num,
                "rows": 0,
                "columns": 0,
                "data": [],
                "text_representation": ""
            }
        
        # Clean up table data
        cleaned_data = []
        for row in table_data:
            cleaned_row = [str(cell).strip() if cell else "" for cell in row]
            cleaned_data.append(cleaned_row)
        
        # Create text representation
        text_lines = []
        for row in cleaned_data:
            text_lines.append(" | ".join(row))
        text_representation = "\n".join(text_lines)
        
        return {
            "page": page_num,
            "table_number": table_num,
            "rows": len(cleaned_data),
            "columns": len(cleaned_data[0]) if cleaned_data else 0,
            "data": cleaned_data,
            "text_representation": text_representation
        }
    
    async def is_scanned_pdf(self, filepath: str) -> bool:
        """Determine if a PDF is scanned (image-based) or text-based."""
        if not OCR_AVAILABLE:
            return False
        
        try:
            with pdfplumber.open(filepath) as pdf:
                # Check first few pages for text content
                text_content = ""
                pages_to_check = min(3, len(pdf.pages))
                
                for page in pdf.pages[:pages_to_check]:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text
                
                # If very little text content, likely scanned
                return len(text_content.strip()) < 100
                
        except Exception as e:
            logger.error(f"Error checking if PDF is scanned: {e}")
            return False
    
    def get_ocr_status(self) -> Dict[str, Any]:
        """Get OCR service status and capabilities."""
        return {
            "available": self.ocr_available,
            "dependencies_installed": OCR_AVAILABLE,
            "capabilities": {
                "pdf_ocr": self.ocr_available,
                "table_extraction": OCR_AVAILABLE,
                "scanned_pdf_detection": OCR_AVAILABLE
            },
            "tesseract_config": getattr(self, 'tesseract_config', None)
        }


# Global OCR service instance
ocr_service = OCRService()
