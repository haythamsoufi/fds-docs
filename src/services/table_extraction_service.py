"""Table extraction service for PDFs using pdfplumber and camelot."""

import asyncio
import io
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import csv

# Optional imports with fallbacks
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False

logger = logging.getLogger(__name__)


class TableExtractionService:
    """Service for extracting tables from PDFs using multiple methods."""
    
    def __init__(self):
        self.pdfplumber_available = PDFPLUMBER_AVAILABLE
        self.camelot_available = CAMELOT_AVAILABLE
        
        if not self.pdfplumber_available and not self.camelot_available:
            logger.warning("No table extraction libraries available. Install pdfplumber or camelot-py[cv] for table extraction.")
    
    async def extract_tables_from_pdf(self, filepath: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Extract all tables from a PDF file using multiple methods."""
        tables = []
        metadata = {
            "extraction_methods_used": [],
            "total_tables_found": 0,
            "pdfplumber_tables": 0,
            "camelot_tables": 0
        }
        
        try:
            # Method 1: pdfplumber (better for simple tables)
            if self.pdfplumber_available:
                pdfplumber_tables = await self._extract_with_pdfplumber(filepath)
                if pdfplumber_tables:
                    tables.extend(pdfplumber_tables)
                    metadata["pdfplumber_tables"] = len(pdfplumber_tables)
                    metadata["extraction_methods_used"].append("pdfplumber")
            
            # Method 2: camelot (better for complex tables with borders)
            if self.camelot_available:
                camelot_tables = await self._extract_with_camelot(filepath)
                if camelot_tables:
                    # Deduplicate tables that might be found by both methods
                    camelot_tables = self._deduplicate_tables(tables, camelot_tables)
                    if camelot_tables:
                        tables.extend(camelot_tables)
                        metadata["camelot_tables"] = len(camelot_tables)
                        metadata["extraction_methods_used"].append("camelot")
            
            metadata["total_tables_found"] = len(tables)
            
            # Sort tables by page number and position
            tables.sort(key=lambda x: (x.get('page', 0), x.get('bbox', [0, 0, 0, 0])[1]))
            
        except Exception as e:
            logger.error(f"Error extracting tables from {filepath}: {e}")
            metadata["error"] = str(e)
        
        return tables, metadata
    
    async def _extract_with_pdfplumber(self, filepath: str) -> List[Dict[str, Any]]:
        """Extract tables using pdfplumber."""
        tables = []
        
        try:
            loop = asyncio.get_event_loop()
            
            def extract_tables():
                pdf_tables = []
                with pdfplumber.open(filepath) as pdf:
                    for page_num, page in enumerate(pdf.pages, 1):
                        page_tables = page.extract_tables()
                        for table_num, table in enumerate(page_tables, 1):
                            if table and len(table) > 1:  # Ensure we have at least a header and one row
                                # Convert table to structured format
                                structured_table = self._structure_pdfplumber_table(
                                    table, page_num, table_num
                                )
                                if structured_table:
                                    pdf_tables.append(structured_table)
                return pdf_tables
            
            tables = await loop.run_in_executor(None, extract_tables)
            
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed for {filepath}: {e}")
        
        return tables
    
    async def _extract_with_camelot(self, filepath: str) -> List[Dict[str, Any]]:
        """Extract tables using camelot."""
        tables = []
        
        try:
            loop = asyncio.get_event_loop()
            
            def extract_tables():
                pdf_tables = []
                # Use 'lattice' method for tables with borders, 'stream' for without
                for method in ['lattice', 'stream']:
                    try:
                        camelot_tables = camelot.read_pdf(
                            filepath, 
                            pages='all', 
                            flavor=method,
                            line_scale=40
                        )
                        
                        for table in camelot_tables:
                            if len(table.df) > 1:  # Ensure we have data
                                structured_table = self._structure_camelot_table(table)
                                if structured_table:
                                    pdf_tables.append(structured_table)
                        
                        if pdf_tables:  # If we found tables with this method, stop
                            break
                            
                    except Exception as method_error:
                        logger.debug(f"Camelot {method} method failed: {method_error}")
                        continue
                
                return pdf_tables
            
            tables = await loop.run_in_executor(None, extract_tables)
            
        except Exception as e:
            logger.warning(f"Camelot extraction failed for {filepath}: {e}")
        
        return tables
    
    def _structure_pdfplumber_table(self, table: List[List[str]], page_num: int, table_num: int) -> Optional[Dict[str, Any]]:
        """Convert pdfplumber table to structured format."""
        try:
            if not table or len(table) < 2:
                return None
            
            # Clean and process table data
            cleaned_table = []
            for row in table:
                if row:
                    cleaned_row = [cell.strip() if cell else "" for cell in row]
                    if any(cell for cell in cleaned_row):  # Skip completely empty rows
                        cleaned_table.append(cleaned_row)
            
            if len(cleaned_table) < 2:
                return None
            
            # Create structured representation
            structured = {
                "table_id": f"pdfplumber_p{page_num}_t{table_num}",
                "page": page_num,
                "table_number": table_num,
                "extraction_method": "pdfplumber",
                "rows": len(cleaned_table),
                "columns": len(cleaned_table[0]) if cleaned_table else 0,
                "data": cleaned_table,
                "headers": cleaned_table[0] if cleaned_table else [],
                "text_representation": self._table_to_text(cleaned_table),
                "csv_representation": self._table_to_csv(cleaned_table),
                "json_representation": self._table_to_json(cleaned_table)
            }
            
            return structured
            
        except Exception as e:
            logger.warning(f"Error structuring pdfplumber table: {e}")
            return None
    
    def _structure_camelot_table(self, table) -> Optional[Dict[str, Any]]:
        """Convert camelot table to structured format."""
        try:
            df = table.df
            if df.empty or len(df) < 2:
                return None
            
            # Convert DataFrame to list of lists
            table_data = df.values.tolist()
            
            # Clean data
            cleaned_table = []
            for row in table_data:
                cleaned_row = [str(cell).strip() if pd.notna(cell) else "" for cell in row]
                if any(cell for cell in cleaned_row):
                    cleaned_table.append(cleaned_row)
            
            if len(cleaned_table) < 2:
                return None
            
            structured = {
                "table_id": f"camelot_p{table.page}_t{table.order}",
                "page": table.page,
                "table_number": table.order,
                "extraction_method": "camelot",
                "accuracy": getattr(table, 'accuracy', None),
                "rows": len(cleaned_table),
                "columns": len(cleaned_table[0]) if cleaned_table else 0,
                "data": cleaned_table,
                "headers": cleaned_table[0] if cleaned_table else [],
                "text_representation": self._table_to_text(cleaned_table),
                "csv_representation": self._table_to_csv(cleaned_table),
                "json_representation": self._table_to_json(cleaned_table),
                "bbox": getattr(table, '_bbox', None)
            }
            
            return structured
            
        except Exception as e:
            logger.warning(f"Error structuring camelot table: {e}")
            return None
    
    def _table_to_text(self, table_data: List[List[str]]) -> str:
        """Convert table data to human-readable text."""
        if not table_data:
            return ""
        
        text_parts = []
        
        # Add header
        if table_data:
            headers = table_data[0]
            text_parts.append("Table with columns: " + " | ".join(headers))
            text_parts.append("")  # Empty line
        
        # Add rows
        for i, row in enumerate(table_data[1:], 1):
            row_text = f"Row {i}: " + " | ".join(row)
            text_parts.append(row_text)
        
        return "\n".join(text_parts)
    
    def _table_to_csv(self, table_data: List[List[str]]) -> str:
        """Convert table data to CSV format."""
        if not table_data:
            return ""
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerows(table_data)
        return output.getvalue()
    
    def _table_to_json(self, table_data: List[List[str]]) -> str:
        """Convert table data to JSON format."""
        if not table_data:
            return "{}"
        
        # Create list of dictionaries using first row as headers
        headers = table_data[0]
        json_data = []
        
        for row in table_data[1:]:
            row_dict = {}
            for i, cell in enumerate(row):
                if i < len(headers):
                    row_dict[headers[i]] = cell
            json_data.append(row_dict)
        
        return json.dumps(json_data, indent=2)
    
    def _deduplicate_tables(self, existing_tables: List[Dict], new_tables: List[Dict]) -> List[Dict]:
        """Remove duplicate tables found by multiple extraction methods."""
        if not existing_tables:
            return new_tables
        
        deduplicated = []
        
        for new_table in new_tables:
            is_duplicate = False
            
            for existing_table in existing_tables:
                # Check if tables are on the same page and have similar content
                if (new_table.get('page') == existing_table.get('page') and
                    self._tables_similar(new_table, existing_table)):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(new_table)
        
        return deduplicated
    
    def _tables_similar(self, table1: Dict, table2: Dict) -> bool:
        """Check if two tables are similar (likely the same table found by different methods)."""
        try:
            # Compare dimensions
            if (table1.get('rows') != table2.get('rows') or 
                table1.get('columns') != table2.get('columns')):
                return False
            
            # Compare headers (first row)
            headers1 = table1.get('headers', [])
            headers2 = table2.get('headers', [])
            
            if len(headers1) != len(headers2):
                return False
            
            # Check if headers are similar (allowing for minor differences)
            similar_headers = 0
            for h1, h2 in zip(headers1, headers2):
                if h1.strip().lower() == h2.strip().lower():
                    similar_headers += 1
                elif abs(len(h1) - len(h2)) <= 2:  # Allow small differences
                    similar_headers += 0.5
            
            # If at least 70% of headers are similar, consider tables similar
            return similar_headers / len(headers1) >= 0.7
            
        except Exception:
            return False


# Global instance
table_extraction_service = TableExtractionService()
