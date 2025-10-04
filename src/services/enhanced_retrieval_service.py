"""Enhanced retrieval service that combines text search with structured data search."""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import re

from .retrieval_service import HybridRetriever, SearchResult, SearchType
from .structured_data_service import structured_data_service
from .embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class EnhancedRetriever:
    """Enhanced retriever that combines text and structured data search."""
    
    def __init__(self, embedding_service: EmbeddingService):
        self.base_retriever = HybridRetriever(embedding_service)
        self.structured_service = structured_data_service
    
    async def retrieve_enhanced(
        self, 
        query: str, 
        k: int = 5,
        search_type: SearchType = SearchType.HYBRID,
        filters: Optional[Dict[str, Any]] = None,
        include_structured_data: bool = True
    ) -> Tuple[List[SearchResult], List[Dict[str, Any]]]:
        """Retrieve both text chunks and structured data for a query."""
        
        # Run base text retrieval
        text_results = await self.base_retriever.retrieve(
            query=query,
            k=k,
            search_type=search_type,
            filters=filters
        )
        
        structured_results = []
        
        if include_structured_data:
            # Determine if this is a numeric query that would benefit from structured data
            if self._is_numeric_query(query):
                structured_results = await self._retrieve_structured_data(query, filters)
        
        return text_results, structured_results
    
    def _is_numeric_query(self, query: str) -> bool:
        """Determine if the query is asking for numeric data."""
        numeric_patterns = [
            r'\b(how many|how much|number of|count of)\b',
            r'\b(total|sum|amount|value|cost|price|budget|funding)\b',
            r'\b(percentage|percent|%)\b',
            r'\b(million|billion|thousand|k|m|b)\b',
            r'\b(compare|difference|ratio|rate)\b'
        ]
        
        query_lower = query.lower()
        for pattern in numeric_patterns:
            if re.search(pattern, query_lower):
                return True
        
        return False
    
    async def _retrieve_structured_data(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant structured data for the query."""
        try:
            # Extract document IDs from filters if available
            document_ids = None
            if filters and 'document_ids' in filters:
                document_ids = filters['document_ids']
            
            # Search for structured data containing query terms
            structured_results = await self.structured_service.search_structured_data(
                query=query,
                document_ids=document_ids
            )
            
            # Also get numeric data if this looks like a numeric query
            numeric_data = []
            if self._is_numeric_query(query):
                numeric_data = await self.structured_service.get_numeric_data(
                    document_id=document_ids[0] if document_ids and len(document_ids) == 1 else None
                )
            
            # Combine and rank results
            combined_results = self._combine_structured_results(structured_results, numeric_data, query)
            
            return combined_results[:10]  # Limit to top 10 structured results
            
        except Exception as e:
            logger.error(f"Error retrieving structured data: {e}")
            return []
    
    def _combine_structured_results(
        self, 
        structured_results: List[Dict[str, Any]], 
        numeric_data: List[Dict[str, Any]], 
        query: str
    ) -> List[Dict[str, Any]]:
        """Combine and rank structured data results."""
        combined = []
        
        # Add structured data results
        for result in structured_results:
            score = self._calculate_structured_score(result, query)
            combined.append({
                "type": "structured_data",
                "content_type": result["content_type"],
                "data": result["data"],
                "metadata": result["metadata"],
                "score": score,
                "document_id": result["document_id"],
                "page_number": result["page_number"],
                "searchable_text": result["searchable_text"]
            })
        
        # Add numeric data results
        for num_data in numeric_data:
            score = self._calculate_numeric_score(num_data, query)
            combined.append({
                "type": "numeric_data",
                "content_type": "numeric",
                "data": num_data,
                "metadata": {
                    "value": num_data["value"],
                    "type": num_data["type"],
                    "context": num_data["context"]
                },
                "score": score,
                "document_id": num_data["document_id"],
                "page_number": num_data["page_number"],
                "searchable_text": num_data["original_text"]
            })
        
        # Sort by score (highest first)
        combined.sort(key=lambda x: x["score"], reverse=True)
        
        return combined
    
    def _calculate_structured_score(self, result: Dict[str, Any], query: str) -> float:
        """Calculate relevance score for structured data."""
        score = 0.0
        searchable_text = result.get("searchable_text", "").lower()
        query_lower = query.lower()
        
        # Exact matches get higher scores
        if query_lower in searchable_text:
            score += 1.0
        
        # Word overlap scoring
        query_words = set(query_lower.split())
        text_words = set(searchable_text.split())
        overlap = len(query_words & text_words)
        
        if query_words:
            score += (overlap / len(query_words)) * 0.5
        
        # Boost for numeric content in numeric queries
        if self._is_numeric_query(query) and result["content_type"] in ["table", "chart"]:
            score += 0.3
        
        # Boost for tables in data-heavy queries
        if any(word in query_lower for word in ["table", "data", "figure", "chart"]) and result["content_type"] == "table":
            score += 0.2
        
        return min(score, 2.0)  # Cap at 2.0
    
    def _calculate_numeric_score(self, num_data: Dict[str, Any], query: str) -> float:
        """Calculate relevance score for numeric data."""
        score = 0.0
        query_lower = query.lower()
        
        # Check if the numeric value type matches query intent
        num_type = num_data["type"]
        original_text = num_data["original_text"].lower()
        
        # Currency queries
        if any(word in query_lower for word in ["dollar", "$", "cost", "price", "budget", "funding"]) and "currency" in num_type:
            score += 1.0
        
        # Percentage queries
        if "%" in query_lower or "percentage" in query_lower or "percent" in query_lower:
            if "percentage" in num_type:
                score += 1.0
        
        # Large number queries
        if any(word in query_lower for word in ["million", "billion", "thousand", "total", "amount"]):
            if "large_number" in num_type:
                score += 1.0
        
        # General numeric queries
        if any(word in query_lower for word in ["number", "count", "how many", "how much"]):
            score += 0.5
        
        # Context matching
        context = num_data.get("context", {})
        if isinstance(context, dict):
            # Check headers for table data
            headers = context.get("headers", [])
            for header in headers:
                if any(word in header.lower() for word in query_lower.split()):
                    score += 0.3
            
            # Check chart elements
            chart_elements = context.get("chart_elements", {})
            for element_list in chart_elements.values():
                for element in element_list:
                    if any(word in element.lower() for word in query_lower.split()):
                        score += 0.2
        
        return min(score, 2.0)  # Cap at 2.0
    
    async def get_document_structured_data(self, document_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """Get all structured data for a specific document."""
        try:
            structured_data = await self.structured_service.get_structured_data(document_id=document_id)
            numeric_data = await self.structured_service.get_numeric_data(document_id=document_id)
            
            return {
                "structured_data": structured_data,
                "numeric_data": numeric_data
            }
            
        except Exception as e:
            logger.error(f"Error getting document structured data: {e}")
            return {"structured_data": [], "numeric_data": []}
    
    async def search_by_numeric_range(
        self, 
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search for numeric data within a specific range."""
        try:
            numeric_data = await self.structured_service.get_numeric_data(document_ids=document_ids)
            
            filtered_data = []
            for num_data in numeric_data:
                value = num_data["value"]
                
                if min_value is not None and value < min_value:
                    continue
                
                if max_value is not None and value > max_value:
                    continue
                
                filtered_data.append(num_data)
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"Error searching by numeric range: {e}")
            return []
