#!/usr/bin/env python3
"""
Example script demonstrating multimodal document processing capabilities.

This script shows how to:
1. Extract tables and charts from a PDF
2. Store structured data
3. Query both text and structured data
4. Generate answers using enhanced LLM prompts
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from services.table_extraction_service import table_extraction_service
from services.chart_extraction_service import chart_extraction_service
from services.structured_data_service import structured_data_service
from services.enhanced_retrieval_service import EnhancedRetriever
from services.embedding_service import EmbeddingService


async def demonstrate_table_extraction(pdf_path: str):
    """Demonstrate table extraction from a PDF."""
    print(f"\n=== Table Extraction from {pdf_path} ===")
    
    try:
        tables, metadata = await table_extraction_service.extract_tables_from_pdf(pdf_path)
        
        print(f"Found {len(tables)} tables")
        print(f"Extraction methods used: {metadata.get('extraction_methods_used', [])}")
        
        for i, table in enumerate(tables, 1):
            print(f"\nTable {i} (Page {table.get('page', 'Unknown')}):")
            print(f"  Rows: {table.get('rows', 0)}")
            print(f"  Columns: {table.get('columns', 0)}")
            print(f"  Method: {table.get('extraction_method', 'Unknown')}")
            
            # Show first few rows
            data = table.get('data', [])
            if data:
                print("  Sample data:")
                for j, row in enumerate(data[:3]):  # Show first 3 rows
                    print(f"    Row {j+1}: {row}")
                if len(data) > 3:
                    print(f"    ... and {len(data)-3} more rows")
            
            # Show text representation
            text_repr = table.get('text_representation', '')
            if text_repr:
                print(f"  Text representation: {text_repr[:100]}...")
        
        return tables
        
    except Exception as e:
        print(f"Error extracting tables: {e}")
        return []


async def demonstrate_chart_extraction(pdf_path: str):
    """Demonstrate chart extraction from a PDF."""
    print(f"\n=== Chart Extraction from {pdf_path} ===")
    
    try:
        charts, metadata = await chart_extraction_service.extract_charts_from_pdf(pdf_path)
        
        print(f"Found {len(charts)} charts/figures")
        print(f"Pages processed: {metadata.get('pages_processed', 0)}")
        
        for i, chart in enumerate(charts, 1):
            print(f"\nChart {i} (Page {chart.get('page', 'Unknown')}):")
            print(f"  Confidence: {chart.get('confidence', 0):.2f}")
            
            # Show numeric values
            numeric_values = chart.get('numeric_values', [])
            if numeric_values:
                print(f"  Numeric values found: {len(numeric_values)}")
                for num_data in numeric_values[:5]:  # Show first 5
                    print(f"    {num_data['original_text']} ({num_data['type']})")
            
            # Show chart elements
            chart_elements = chart.get('chart_elements', {})
            if chart_elements:
                print("  Chart elements:")
                for element_type, elements in chart_elements.items():
                    if elements:
                        print(f"    {element_type}: {elements[:3]}")  # Show first 3
            
            # Show text representation
            text_repr = chart.get('text_representation', '')
            if text_repr:
                print(f"  Text representation: {text_repr[:100]}...")
        
        return charts
        
    except Exception as e:
        print(f"Error extracting charts: {e}")
        return []


async def demonstrate_structured_data_storage(tables, charts, document_id: str):
    """Demonstrate storing structured data."""
    print(f"\n=== Storing Structured Data for Document {document_id} ===")
    
    try:
        # Store tables
        if tables:
            table_ids = await structured_data_service.store_structured_data(
                document_id=document_id,
                structured_items=tables,
                content_type="table"
            )
            print(f"Stored {len(table_ids)} tables: {table_ids}")
        
        # Store charts
        if charts:
            chart_ids = await structured_data_service.store_structured_data(
                document_id=document_id,
                structured_items=charts,
                content_type="chart"
            )
            print(f"Stored {len(chart_ids)} charts: {chart_ids}")
        
        # Get statistics
        stats = await structured_data_service.get_statistics()
        print(f"\nDatabase statistics:")
        print(f"  Total items: {stats.get('total_items', 0)}")
        print(f"  By content type: {stats.get('by_content_type', {})}")
        print(f"  Documents with structured data: {stats.get('documents_with_structured_data', 0)}")
        
    except Exception as e:
        print(f"Error storing structured data: {e}")


async def demonstrate_enhanced_retrieval(queries: list):
    """Demonstrate enhanced retrieval with structured data."""
    print(f"\n=== Enhanced Retrieval Demo ===")
    
    try:
        # Initialize embedding service and enhanced retriever
        embedding_service = EmbeddingService()
        await embedding_service.initialize()
        enhanced_retriever = EnhancedRetriever(embedding_service)
        
        for query in queries:
            print(f"\nQuery: '{query}'")
            
            # Perform enhanced retrieval
            text_results, structured_results = await enhanced_retriever.retrieve_enhanced(
                query=query,
                k=5,
                include_structured_data=True
            )
            
            print(f"  Text results: {len(text_results)}")
            print(f"  Structured results: {len(structured_results)}")
            
            # Show structured results
            for i, struct_result in enumerate(structured_results[:3], 1):
                content_type = struct_result.get('content_type', 'unknown')
                score = struct_result.get('score', 0)
                print(f"    {i}. {content_type} (score: {score:.2f})")
                
                if content_type == "numeric":
                    data = struct_result.get('data', {})
                    print(f"       Value: {data.get('value')} ({data.get('original_text')})")
                elif content_type in ["table", "chart"]:
                    data = struct_result.get('data', {})
                    text_repr = data.get('text_representation', '')
                    if text_repr:
                        print(f"       Summary: {text_repr[:80]}...")
        
    except Exception as e:
        print(f"Error in enhanced retrieval: {e}")


async def main():
    """Main demonstration function."""
    print("=== Multimodal Document Processing Demo ===\n")
    
    # Example PDF path (replace with your PDF)
    pdf_path = "data/documents/Syria_INP_AR_2024.pdf"
    
    if not Path(pdf_path).exists():
        print(f"PDF file not found: {pdf_path}")
        print("Please update the pdf_path variable with a valid PDF file.")
        return
    
    # Document ID for storage
    document_id = "demo_document_001"
    
    # Step 1: Extract tables
    tables = await demonstrate_table_extraction(pdf_path)
    
    # Step 2: Extract charts
    charts = await demonstrate_chart_extraction(pdf_path)
    
    # Step 3: Store structured data
    await demonstrate_structured_data_storage(tables, charts, document_id)
    
    # Step 4: Demonstrate enhanced retrieval
    sample_queries = [
        "How much funding was allocated?",
        "What are the key numbers in the document?",
        "Show me the budget breakdown",
        "What data is in the tables?",
        "Extract numeric values from charts"
    ]
    
    await demonstrate_enhanced_retrieval(sample_queries)
    
    print("\n=== Demo Complete ===")
    print("The multimodal processing system is now ready to handle:")
    print("• Table extraction and structured storage")
    print("• Chart extraction with OCR")
    print("• Enhanced retrieval combining text and structured data")
    print("• LLM integration with multimodal context")


if __name__ == "__main__":
    asyncio.run(main())
