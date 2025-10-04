"""Chart and figure extraction service for PDFs using OCR and image analysis."""

import asyncio
import io
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import re

# Optional imports with fallbacks
try:
    from PIL import Image
    import pdf2image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

logger = logging.getLogger(__name__)


class ChartExtractionService:
    """Service for extracting charts, figures, and numeric data from PDFs."""
    
    def __init__(self):
        self.pil_available = PIL_AVAILABLE
        self.easyocr_available = EASYOCR_AVAILABLE
        self.opencv_available = OPENCV_AVAILABLE
        
        self.ocr_reader = None
        if self.easyocr_available:
            try:
                # Initialize OCR reader (CPU-only for Docker compatibility)
                self.ocr_reader = easyocr.Reader(['en'], gpu=False)
                logger.info("EasyOCR initialized for chart extraction")
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {e}")
                self.easyocr_available = False
        
        if not self.pil_available:
            logger.warning("PIL not available. Chart extraction will be limited.")
    
    async def extract_charts_from_pdf(self, filepath: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Extract charts, figures, and numeric data from a PDF file."""
        charts = []
        metadata = {
            "extraction_methods_used": [],
            "total_charts_found": 0,
            "pages_processed": 0,
            "ocr_extractions": 0
        }
        
        try:
            if not self.pil_available:
                logger.warning("PIL not available for chart extraction")
                return charts, metadata
            
            # Convert PDF to images
            images = await self._pdf_to_images(filepath)
            metadata["pages_processed"] = len(images)
            
            for page_num, image in enumerate(images, 1):
                page_charts = await self._extract_charts_from_image(image, page_num)
                if page_charts:
                    charts.extend(page_charts)
                    metadata["ocr_extractions"] += len(page_charts)
            
            metadata["total_charts_found"] = len(charts)
            metadata["extraction_methods_used"] = ["ocr", "image_analysis"]
            
            # Sort charts by page number
            charts.sort(key=lambda x: x.get('page', 0))
            
        except Exception as e:
            logger.error(f"Error extracting charts from {filepath}: {e}")
            metadata["error"] = str(e)
        
        return charts, metadata
    
    async def _pdf_to_images(self, filepath: str, dpi: int = 300) -> List[Image.Image]:
        """Convert PDF pages to images."""
        images = []
        
        try:
            loop = asyncio.get_event_loop()
            
            def convert_pdf():
                return pdf2image.convert_from_path(filepath, dpi=dpi)
            
            images = await loop.run_in_executor(None, convert_pdf)
            
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
        
        return images
    
    async def _extract_charts_from_image(self, image: Image.Image, page_num: int) -> List[Dict[str, Any]]:
        """Extract charts and numeric data from a single image."""
        charts = []
        
        try:
            # Detect potential chart regions
            chart_regions = await self._detect_chart_regions(image)
            
            for region_num, region in enumerate(chart_regions, 1):
                # Extract text from the region using OCR
                ocr_data = await self._extract_text_from_region(image, region)
                
                if ocr_data and self._is_chart_like(ocr_data):
                    # Process the extracted data
                    structured_chart = self._structure_chart_data(
                        ocr_data, page_num, region_num, region
                    )
                    
                    if structured_chart:
                        charts.append(structured_chart)
            
        except Exception as e:
            logger.warning(f"Error extracting charts from page {page_num}: {e}")
        
        return charts
    
    async def _detect_chart_regions(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect regions in the image that likely contain charts or figures."""
        regions = []
        
        try:
            if not self.opencv_available:
                # Fallback: treat entire image as one region
                width, height = image.size
                regions.append({
                    "bbox": [0, 0, width, height],
                    "confidence": 0.5,
                    "type": "full_page"
                })
                return regions
            
            loop = asyncio.get_event_loop()
            
            def detect_regions():
                # Convert PIL image to OpenCV format
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Simple region detection based on contours and text density
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                
                # Find contours
                contours, _ = cv2.findContours(
                    cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1],
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                
                detected_regions = []
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Filter regions by size (charts are usually reasonably large)
                    if w > 100 and h > 100:
                        detected_regions.append({
                            "bbox": [x, y, x + w, y + h],
                            "confidence": 0.7,
                            "type": "contour_detected"
                        })
                
                return detected_regions
            
            regions = await loop.run_in_executor(None, detect_regions)
            
            # If no regions detected, use the full page
            if not regions:
                width, height = image.size
                regions.append({
                    "bbox": [0, 0, width, height],
                    "confidence": 0.3,
                    "type": "full_page_fallback"
                })
            
        except Exception as e:
            logger.warning(f"Error detecting chart regions: {e}")
            # Fallback to full page
            width, height = image.size
            regions.append({
                "bbox": [0, 0, width, height],
                "confidence": 0.2,
                "type": "error_fallback"
            })
        
        return regions
    
    async def _extract_text_from_region(self, image: Image.Image, region: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract text from a specific region using OCR."""
        if not self.easyocr_available or not self.ocr_reader:
            return None
        
        try:
            # Crop the region from the image
            bbox = region["bbox"]
            cropped_image = image.crop(bbox)
            
            loop = asyncio.get_event_loop()
            
            def run_ocr():
                # Convert PIL image to numpy array for EasyOCR
                img_array = np.array(cropped_image)
                results = self.ocr_reader.readtext(img_array)
                return results
            
            ocr_results = await loop.run_in_executor(None, run_ocr)
            
            if not ocr_results:
                return None
            
            # Process OCR results
            extracted_text = []
            coordinates = []
            
            for (bbox_coords, text, confidence) in ocr_results:
                if confidence > 0.5:  # Filter low-confidence results
                    extracted_text.append(text)
                    coordinates.append(bbox_coords)
            
            if not extracted_text:
                return None
            
            return {
                "text": extracted_text,
                "coordinates": coordinates,
                "full_text": " ".join(extracted_text),
                "confidence_scores": [conf for _, _, conf in ocr_results if conf > 0.5]
            }
            
        except Exception as e:
            logger.warning(f"Error extracting text from region: {e}")
            return None
    
    def _is_chart_like(self, ocr_data: Dict[str, Any]) -> bool:
        """Determine if the extracted text looks like it's from a chart or figure."""
        if not ocr_data:
            return False
        
        text = ocr_data.get("full_text", "").lower()
        
        # Look for chart-like patterns
        chart_indicators = [
            # Numbers and percentages
            r'\d+%', r'\d+\.\d+', r'\$\d+', r'\d+,\d+',
            # Chart-related words
            r'\b(chart|graph|figure|table|bar|line|pie)\b',
            r'\b(funding|budget|expenditure|revenue|cost)\b',
            r'\b(million|billion|thousand|k|m|b)\b',
            # Axis labels
            r'\b(x|y|axis|horizontal|vertical)\b',
            # Common chart elements
            r'\b(legend|title|caption|source)\b'
        ]
        
        matches = 0
        for pattern in chart_indicators:
            if re.search(pattern, text):
                matches += 1
        
        # If we have multiple indicators or lots of numbers, likely a chart
        return matches >= 2 or len(re.findall(r'\d+', text)) >= 3
    
    def _structure_chart_data(self, ocr_data: Dict[str, Any], page_num: int, region_num: int, region: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Structure the extracted chart data into a standardized format."""
        try:
            full_text = ocr_data.get("full_text", "")
            text_list = ocr_data.get("text", [])
            
            # Extract numeric values
            numeric_values = self._extract_numeric_values(full_text)
            
            # Extract potential chart elements
            chart_elements = self._extract_chart_elements(text_list)
            
            # Create structured representation
            structured = {
                "chart_id": f"chart_p{page_num}_r{region_num}",
                "page": page_num,
                "region_number": region_num,
                "extraction_method": "ocr",
                "bbox": region.get("bbox"),
                "confidence": region.get("confidence", 0.5),
                "raw_text": full_text,
                "text_elements": text_list,
                "numeric_values": numeric_values,
                "chart_elements": chart_elements,
                "text_representation": self._chart_to_text(numeric_values, chart_elements),
                "json_representation": json.dumps({
                    "numeric_data": numeric_values,
                    "chart_elements": chart_elements,
                    "raw_text": full_text
                }, indent=2)
            }
            
            return structured
            
        except Exception as e:
            logger.warning(f"Error structuring chart data: {e}")
            return None
    
    def _extract_numeric_values(self, text: str) -> List[Dict[str, Any]]:
        """Extract numeric values and their context from text."""
        numeric_values = []
        
        # Patterns for different types of numbers
        patterns = [
            # Currency
            (r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', 'currency'),
            # Percentages
            (r'(\d+(?:\.\d+)?)\s*%', 'percentage'),
            # Large numbers with units
            (r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(million|billion|thousand|m|b|k)', 'large_number'),
            # Regular numbers
            (r'\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\b', 'number')
        ]
        
        for pattern, number_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value = match.group(1).replace(',', '')
                try:
                    numeric_value = float(value)
                    numeric_values.append({
                        "value": numeric_value,
                        "type": number_type,
                        "original_text": match.group(0),
                        "position": match.start()
                    })
                except ValueError:
                    continue
        
        return numeric_values
    
    def _extract_chart_elements(self, text_list: List[str]) -> Dict[str, List[str]]:
        """Extract potential chart elements (titles, labels, etc.) from text."""
        elements = {
            "titles": [],
            "labels": [],
            "units": [],
            "categories": []
        }
        
        for text in text_list:
            text_lower = text.lower().strip()
            
            # Titles (usually longer text at the beginning)
            if len(text) > 10 and any(word in text_lower for word in ['chart', 'graph', 'figure', 'analysis']):
                elements["titles"].append(text)
            
            # Units
            elif text_lower in ['million', 'billion', 'thousand', '%', '$', 'usd', 'eur']:
                elements["units"].append(text)
            
            # Categories (shorter labels)
            elif len(text) < 20 and not re.search(r'\d', text):
                elements["categories"].append(text)
            
            # General labels
            else:
                elements["labels"].append(text)
        
        return elements
    
    def _chart_to_text(self, numeric_values: List[Dict[str, Any]], chart_elements: Dict[str, List[str]]) -> str:
        """Convert chart data to human-readable text."""
        text_parts = []
        
        # Add title if available
        if chart_elements.get("titles"):
            text_parts.append(f"Chart: {chart_elements['titles'][0]}")
            text_parts.append("")
        
        # Add numeric data
        if numeric_values:
            text_parts.append("Key numeric values:")
            for num_data in numeric_values:
                text_parts.append(f"- {num_data['original_text']}")
            text_parts.append("")
        
        # Add categories if available
        if chart_elements.get("categories"):
            text_parts.append(f"Categories: {', '.join(chart_elements['categories'])}")
        
        return "\n".join(text_parts)


# Global instance
chart_extraction_service = ChartExtractionService()
