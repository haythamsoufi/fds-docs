# Citation and Numeric Answer System

This document describes the enhanced citation system and numeric answer capabilities implemented in the FDS RAG system.

## Overview

The system now provides two key enhancements for better query responses:

1. **Automatic Citation Generation**: Extracts and formats citations from retrieved chunks
2. **Numeric Answer Enhancement**: Provides strict numeric answers for year-based count queries

## Citation System

### How It Works

The citation system automatically generates citations from the top retrieved chunks during query processing:

1. **Retrieval**: System retrieves relevant chunks using hybrid search
2. **Citation Extraction**: Top 3 chunks are converted to citation format
3. **Metadata Enrichment**: Citations include document titles, page numbers, sections
4. **UI Display**: Citations are displayed in the UI with source information

### Citation Format

```json
{
  "id": "chunk_123",
  "document_id": "doc_456", 
  "document_title": "Annual Budget Report 2024",
  "page_number": 15,
  "section_title": "Emergency Funding",
  "chunk_id": "chunk_123",
  "score": 0.95,
  "content": "Emergency response funding totaled $2.5M...",
  "metadata": {
    "page_start": 15,
    "page_end": 15,
    "section_title": "Emergency Funding"
  }
}
```

### API Response

The `/api/v1/query` endpoint now returns citations in the response:

```json
{
  "query": "How much funding was allocated?",
  "answer": "Based on the available sources, $2.5M was allocated for emergency response [Source 1]...",
  "retrieved_chunks": [...],
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
  ],
  "response_time": 1.23,
  "confidence": 0.89
}
```

## Numeric Answer Enhancement

### Year-Based Query Detection

The system automatically detects year-based numeric queries using regex patterns:

- Pattern: `\b(20\d{2}|19\d{2})\b` - Matches years 1900-2099
- Numeric patterns: `\b(how many|number of|count of)\b`
- Example: "In 2024, how many emergency operations were active?"

### Structured Data Integration

For year-based queries, the system:

1. **Extracts Target Year**: Parses the year from the query
2. **Searches Structured Data**: Looks for numeric values in structured data
3. **Year Context Matching**: Matches structured data containing the target year
4. **Numeric Extraction**: Extracts the first plausible integer found
5. **Answer Enhancement**: Prepends the number with citation tag

### Example Implementation

```python
# Year-based numeric query processing
year_match = re.search(r"\b(20\d{2}|19\d{2})\b", normalized)
is_how_many = bool(re.search(r"\b(how many|number of|count of)\b", normalized))

if is_how_many and year_match and structured_results:
    target_year = year_match.group(1)
    # Look for numeric structured entries related to operations in the target year
    for s in structured_results:
        data = s.get("data", {}) or {}
        text_repr = data.get("text_representation", "")
        if target_year in text_repr.lower():
            # Extract first plausible integer
            m = re.search(r"\b(\d{1,6})\b", text_repr)
            if m:
                strict_numeric_answer = m.group(1)
                break
```

### Response Format

For year-based numeric queries, the response format is:

```
{number} [Source 1]

{full_llm_generated_answer_with_context}
```

Example:
```
12 [Source 1]

Based on the available sources, there were 12 active emergency operations in 2024. This information is derived from the annual operations report which tracks all active humanitarian operations throughout the year...
```

## Configuration

### Environment Variables

No additional configuration is required - citations and numeric answers are enabled by default.

### Optional Settings

```env
# Confidence thresholds (existing)
NO_ANSWER_THRESHOLD=0.3
CONFIDENCE_CALIBRATION_ENABLED=true
MIN_CHUNK_SCORE_THRESHOLD=0.2

# Numeric query detection (automatic)
# Year patterns are built-in
# Numeric patterns are built-in
```

## Usage Examples

### Basic Citation Query

```bash
curl -X POST "http://localhost:8080/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key priorities for 2025?",
    "max_results": 10
  }'
```

Response includes citations:
```json
{
  "answer": "The key priorities for 2025 include emergency preparedness, climate adaptation, and community resilience [Source 1]...",
  "citations": [
    {
      "document_title": "Strategic Plan 2025",
      "page_number": 3,
      "section_title": "Key Priorities",
      "content": "Our 2025 priorities focus on emergency preparedness...",
      "score": 0.92
    }
  ]
}
```

### Numeric Query with Year

```bash
curl -X POST "http://localhost:8080/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "In 2024, how many emergency operations were active?",
    "max_results": 20,
    "filters": {
      "search_type": "hybrid"
    }
  }'
```

Response includes strict numeric answer:
```json
{
  "answer": "12 [Source 1]\n\nBased on the available sources, there were 12 active emergency operations in 2024...",
  "citations": [...]
}
```

## UI Integration

### Citation Display

The UI automatically displays citations when available:

- **Citation Block**: Shows source information with page numbers
- **Expandable Content**: Full citation details on demand
- **Source Links**: Direct links to document sections
- **Relevance Scores**: Visual indicators of source relevance

### Numeric Answer Display

For numeric queries, the UI:

- **Highlights Numbers**: Emphasizes the extracted numeric value
- **Shows Citations**: Displays source information for the number
- **Provides Context**: Shows the full explanation below the number

## Troubleshooting

### Citations Not Appearing

1. **Check Retrieved Chunks**: Ensure chunks are being retrieved
2. **Verify Metadata**: Check that document metadata includes titles
3. **Review Logs**: Look for citation generation errors

```bash
# Check if citations are being generated
curl -X POST "http://localhost:8080/api/v1/query" \
  -d '{"query": "test query"}' | jq '.citations'
```

### Numeric Answers Not Working

1. **Check Structured Data**: Ensure structured data exists
2. **Verify Year Detection**: Test year pattern matching
3. **Review Structured Results**: Check if structured data contains target year

```bash
# Check structured data statistics
curl "http://localhost:8080/api/v1/admin/structured-data-stats"
```

### Performance Issues

1. **Limit Citations**: Citations are limited to top 3 chunks
2. **Optimize Retrieval**: Use appropriate max_results setting
3. **Monitor Response Time**: Check query processing time

## Best Practices

### Query Formulation

For best results with numeric queries:

- **Include Year**: "In 2024, how many operations were active?"
- **Be Specific**: "How many emergency operations in Syria in 2024?"
- **Use Count Terms**: "number of", "how many", "count of"

### Citation Quality

To improve citation quality:

- **Use Hybrid Search**: Combines semantic and keyword search
- **Increase Max Results**: More chunks = better citation selection
- **Check Document Quality**: Ensure documents have good metadata

### Performance Optimization

- **Cache Queries**: Enable query caching for repeated questions
- **Limit Citations**: System automatically limits to top 3
- **Monitor Resources**: Watch memory usage with large document sets

## Future Enhancements

### Planned Features

1. **Citation Ranking**: Rank citations by relevance and quality
2. **Cross-Reference Detection**: Link related citations
3. **Citation Summarization**: Auto-generate citation summaries
4. **Advanced Numeric Extraction**: Support for ranges, percentages, ratios

### API Improvements

1. **Citation Filtering**: Filter citations by document type, date, etc.
2. **Numeric Validation**: Validate extracted numbers against known ranges
3. **Citation Analytics**: Track citation usage and effectiveness
