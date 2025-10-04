# Multimodal Document Processing Guide

This guide explains how to use the enhanced document processing capabilities that can extract and query tables, charts, and numeric data from PDFs.

## Overview

The system now supports three types of content extraction:

1. **Text Content** - Traditional text extraction from PDFs
2. **Table Data** - Structured table extraction with multiple methods
3. **Chart/Figure Data** - OCR-based extraction of numeric data from charts and figures

## Features

### Table Extraction
- **pdfplumber**: Excellent for simple tables with clear borders
- **camelot**: Better for complex tables with borders (optional dependency)
- Automatic deduplication when both methods find the same table
- Structured storage with CSV, JSON, and text representations

### Chart Extraction
- **OCR-based**: Uses EasyOCR to extract text from chart images
- **Image Analysis**: Converts PDF pages to images for analysis
- **Numeric Data Extraction**: Identifies and extracts numeric values with context
- **Chart Element Detection**: Identifies titles, labels, units, and categories

### Enhanced Retrieval
- **Hybrid Search**: Combines text search with structured data search
- **Numeric Query Detection**: Automatically identifies queries asking for numbers
- **Contextual Scoring**: Ranks structured data based on query relevance
- **Multimodal Answer Generation**: LLM can analyze both text and structured data
- **Citation Generation**: Automatic citation extraction from retrieved chunks
- **Numeric Answer Path**: Strict numeric responses for year-based count queries

## Installation

### Required Dependencies
Most dependencies are already included in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Optional Dependencies

For enhanced table extraction with camelot:
```bash
pip install camelot-py[cv]
```

Note: camelot requires additional system dependencies (Tkinter, ghostscript). See [camelot documentation](https://camelot-py.readthedocs.io/en/master/user/install.html) for details.

## Configuration

### Environment Variables

Add these to your `.env` file for optimal multimodal processing:

```env
# Enable multimodal processing
MULTIMODAL_PROCESSING=true

# Table extraction settings
TABLE_EXTRACTION_METHODS=pdfplumber,camelot
MIN_TABLE_ROWS=2
MIN_TABLE_COLS=2

# Chart extraction settings
CHART_EXTRACTION_DPI=300
CHART_CONFIDENCE_THRESHOLD=0.5
OCR_LANGUAGES=en,ar

# Enhanced retrieval settings
ENHANCED_RETRIEVAL=true
STRUCTURED_DATA_SEARCH=true
NUMERIC_QUERY_THRESHOLD=0.7
```

## Usage

### Document Processing

The system automatically detects and processes tables and charts when uploading PDFs:

```python
# The multimodal processor is used automatically
from src.services.multimodal_document_processor import multimodal_document_processor

# Process a document (called automatically by the API)
document_id = await multimodal_document_processor.process_document_multimodal("path/to/document.pdf")
```

### Querying with Enhanced Retrieval

The enhanced retrieval system automatically detects numeric queries and searches both text and structured data:

```python
from src.services.enhanced_retrieval_service import EnhancedRetriever

# Initialize enhanced retriever
enhanced_retriever = EnhancedRetriever(embedding_service)

# Query with automatic structured data inclusion
text_results, structured_results = await enhanced_retriever.retrieve_enhanced(
    query="How much funding was allocated?",
    k=10,
    include_structured_data=True
)
```

### Direct Structured Data Access

Access structured data directly:

```python
from src.services.structured_data_service import structured_data_service

# Get all tables from a document
tables = await structured_data_service.get_structured_data(
    document_id="doc_123",
    content_type="table"
)

# Search for specific numeric data
numeric_data = await structured_data_service.search_structured_data(
    query="funding budget",
    content_types=["table", "chart"]
)

# Get numeric values within a range
values = await structured_data_service.get_numeric_data(
    min_value=1000000,
    max_value=50000000
)
```

## Query Examples

### Numeric Queries
These queries automatically trigger structured data search and citation generation:

- "How much funding was allocated?"
- "What is the total budget?"
- "How many people were affected?"
- "What percentage of the budget was spent?"
- "Compare the funding between projects"
- "In 2024, how many emergency operations were active?" (Year-based numeric queries)

### Table-Specific Queries
- "Show me the budget breakdown table"
- "What data is in the funding table?"
- "Extract the numeric values from the tables"

### Chart-Specific Queries
- "What do the charts show about funding?"
- "Extract numbers from the bar chart"
- "What are the key metrics in the figures?"

## Data Storage

### Structured Data Database
Tables and charts are stored in a separate SQLite database (`structured_data.db`) with the following schema:

```sql
CREATE TABLE structured_data (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    content_type TEXT NOT NULL,  -- 'table' or 'chart'
    page_number INTEGER,
    data_json TEXT NOT NULL,     -- Full structured data
    searchable_text TEXT,        -- Text for search indexing
    metadata_json TEXT,          -- Extraction metadata
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

### Data Formats

#### Table Data
```json
{
  "table_id": "pdfplumber_p1_t1",
  "page": 1,
  "rows": 5,
  "columns": 3,
  "headers": ["Category", "Amount", "Percentage"],
  "data": [
    ["Emergency Response", "$2.5M", "45%"],
    ["Logistics", "$1.8M", "32%"],
    ["Administration", "$1.2M", "23%"]
  ],
  "text_representation": "Table with columns: Category | Amount | Percentage...",
  "csv_representation": "Category,Amount,Percentage\nEmergency Response,$2.5M,45%...",
  "json_representation": "[{\"Category\": \"Emergency Response\", \"Amount\": \"$2.5M\"...}]"
}
```

#### Chart Data
```json
{
  "chart_id": "chart_p2_r1",
  "page": 2,
  "numeric_values": [
    {
      "value": 2500000,
      "type": "currency",
      "original_text": "$2.5M",
      "position": 45
    }
  ],
  "chart_elements": {
    "titles": ["Emergency Funding Allocation"],
    "labels": ["Emergency Response", "Logistics", "Administration"],
    "units": ["M", "$", "%"]
  },
  "text_representation": "Chart: Emergency Funding Allocation\nKey numeric values:\n- $2.5M\n- $1.8M..."
}
```

## Performance Considerations

### CPU Usage
- Table extraction is relatively fast
- Chart extraction (OCR) is more CPU-intensive
- Processing time scales with document complexity

### Memory Usage
- PDF to image conversion requires significant memory
- Large tables are stored efficiently in JSON format
- Consider processing documents in batches for large collections

### Optimization Tips
1. **Disable chart extraction** for text-heavy documents by setting `CHART_EXTRACTION_ENABLED=false`
2. **Use pdfplumber only** for simple tables by setting `TABLE_EXTRACTION_METHODS=pdfplumber`
3. **Lower OCR DPI** for faster processing: `CHART_EXTRACTION_DPI=200`
4. **Batch processing** for multiple documents

## Troubleshooting

### Common Issues

#### Table Extraction Not Working
- Check if pdfplumber is installed: `pip install pdfplumber`
- For camelot issues, see [installation guide](https://camelot-py.readthedocs.io/en/master/user/install.html)
- Verify PDF is not image-based (scanned)

#### Chart Extraction Issues
- Ensure PIL and pdf2image are installed
- Check OCR language support: `OCR_LANGUAGES=en,ar`
- Verify PDF pages can be converted to images

#### Poor Numeric Query Results
- Check if structured data was extracted: query the structured data service
- Verify numeric query detection patterns
- Ensure LLM can access structured data in prompts

### Debugging

Enable debug logging:
```python
import logging
logging.getLogger("src.services.table_extraction_service").setLevel(logging.DEBUG)
logging.getLogger("src.services.chart_extraction_service").setLevel(logging.DEBUG)
```

Check structured data statistics:
```python
stats = await structured_data_service.get_statistics()
print(f"Total structured items: {stats['total_items']}")
print(f"By type: {stats['by_content_type']}")
```

## Citation System

### Automatic Citation Generation
The system automatically generates citations from the top retrieved chunks:

```json
{
  "query": "How much funding was allocated?",
  "answer": "Based on the available sources, $2.5M was allocated for emergency response [Source 1]...",
  "citations": [
    {
      "id": "chunk_123",
      "document_id": "doc_456",
      "document_title": "Annual Budget Report 2024",
      "page_number": 15,
      "section_title": "Emergency Funding",
      "chunk_id": "chunk_123",
      "score": 0.95,
      "content": "Emergency response funding totaled $2.5M...",
      "metadata": {"page_start": 15, "page_end": 15}
    }
  ]
}
```

### Numeric Answer Enhancement
For year-based numeric queries, the system provides strict numeric answers:

```json
{
  "query": "In 2024, how many emergency operations were active?",
  "answer": "12 [Source 1]\n\nBased on the available sources, there were 12 active emergency operations in 2024...",
  "citations": [...]
}
```

## API Integration

### Query Endpoint
The `/api/v1/query` endpoint automatically uses enhanced retrieval with citations:

```bash
curl -X POST "http://localhost:8080/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How much funding was allocated?",
    "max_results": 10,
    "filters": {
      "search_type": "hybrid"
    }
  }'
```

### Admin Endpoints
Re-process documents with multimodal extraction:

```bash
# Re-embed with structured data
curl -X POST "http://localhost:8080/api/v1/admin/reembed" \
  -H "Content-Type: application/json" \
  -d '{"force": true, "version": 2}'

# Re-index documents
curl -X POST "http://localhost:8080/api/v1/admin/reindex" \
  -H "Content-Type: application/json" \
  -d '{"force_reprocess": true}'
```

## Future Enhancements

Planned improvements:

1. **Multimodal LLM Integration**: Direct chart/image analysis with vision models
2. **Advanced Chart Recognition**: Better detection of chart types and elements
3. **Cross-Document Analysis**: Compare data across multiple documents
4. **Interactive Visualizations**: Generate charts from extracted data
5. **Export Capabilities**: Export structured data to Excel, CSV, JSON

## Support

For issues with multimodal processing:

1. Check the logs for extraction errors
2. Verify dependencies are installed correctly
3. Test with simple documents first
4. Review the configuration settings
5. Check the structured data database for extracted content
