"""Advanced retrieval service with hybrid search and reranking."""

import asyncio
import re
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_, and_
import numpy as np

from src.core.models import ChunkModel, DocumentModel, QueryIntent
from src.core.config import settings
from src.core.database import get_db_session
from src.core.cache import cache
from src.services.embedding_service import EmbeddingService, VectorStore
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class SearchType(str, Enum):
    """Search type enumeration."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


@dataclass
class SearchResult:
    """Search result with metadata."""
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    document_metadata: Dict[str, Any]


class QueryProcessor:
    """Process and understand user queries."""
    
    def __init__(self):
        self.intent_patterns = {
            QueryIntent.FACTUAL: [
                r'\bwhat\s+is\b', r'\bwho\s+is\b', r'\bwhen\s+did\b',
                r'\bwhere\s+is\b', r'\bhow\s+many\b', r'\bdefine\b'
            ],
            QueryIntent.COMPARISON: [
                r'\bcompare\b', r'\bdifference\b', r'\bversus\b', r'\bvs\b',
                r'\bbetter\b', r'\bworse\b', r'\bsimilar\b'
            ],
            QueryIntent.ANALYTICAL: [
                r'\banalyze\b', r'\bevaluate\b', r'\bassess\b', r'\bexamine\b',
                r'\bwhy\b', r'\bhow\s+does\b', r'\bimpact\b', r'\beffect\b'
            ],
            QueryIntent.PROCEDURAL: [
                r'\bhow\s+to\b', r'\bsteps\b', r'\bprocess\b', r'\bprocedure\b',
                r'\binstructions\b', r'\bguide\b'
            ]
        }
        
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process query and extract intent and entities."""
        query_lower = query.lower()
        
        # Classify intent
        intent = self._classify_intent(query_lower)
        
        # Extract entities (simplified)
        entities = self._extract_entities(query)
        
        # Generate query variations
        variations = self._generate_query_variations(query)
        
        return {
            "original_query": query,
            "intent": intent,
            "entities": entities,
            "variations": variations,
            "processed_query": self._preprocess_query(query)
        }
        
    def _classify_intent(self, query: str) -> QueryIntent:
        """Classify query intent."""
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    return intent
        return QueryIntent.UNKNOWN
        
    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from query (simplified)."""
        # This is a simplified implementation
        # In production, you'd use spaCy, NLTK, or similar
        entities = []
        
        # Extract capitalized words (potential proper nouns)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', query)
        entities.extend(capitalized_words)
        
        # Extract numbers and dates
        numbers = re.findall(r'\b\d{4}\b|\b\d+\b', query)
        entities.extend(numbers)
        
        return list(set(entities))
        
    def _generate_query_variations(self, query: str) -> List[str]:
        """Generate query variations for better retrieval."""
        variations = [query]
        
        # Add variations with synonyms (simplified)
        synonyms = {
            'document': ['file', 'paper', 'report'],
            'policy': ['procedure', 'guideline', 'rule'],
            'strategy': ['plan', 'approach', 'method']
        }
        
        for word, syns in synonyms.items():
            if word in query.lower():
                for syn in syns:
                    variation = re.sub(
                        r'\b' + word + r'\b', 
                        syn, 
                        query, 
                        flags=re.IGNORECASE
                    )
                    if variation != query:
                        variations.append(variation)
                        
        return variations[:5]  # Limit variations
        
    def _preprocess_query(self, query: str) -> str:
        """Preprocess query for better matching."""
        # Remove stop words, normalize, etc.
        # This is simplified - in production use proper NLP preprocessing
        processed = query.lower().strip()
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = processed.split()
        filtered_words = [word for word in words if word not in stop_words]
        
        return ' '.join(filtered_words)


class KeywordSearcher:
    """Keyword-based search using database full-text search."""
    
    async def search(
        self, 
        query: str, 
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform keyword search scored by BM25 over candidate chunks."""
        # For trivial short queries, avoid noisy keyword search
        if len(query.strip()) < 3:
            return []
        async with get_db_session() as session:
            # Tokenize query terms
            query_terms = self._tokenize(query)
            if not query_terms:
                return []
            
            # Build candidate set using LIKE to reduce volume
            search_conditions = []
            for term in query_terms:
                search_conditions.append(ChunkModel.content.ilike(f'%{term}%'))
            search_condition = or_(*search_conditions)
            
            base_query = select(ChunkModel, DocumentModel).join(
                DocumentModel, ChunkModel.document_id == DocumentModel.id
            ).where(search_condition)
            
            if filters:
                base_query = self._apply_filters(base_query, filters)
                
            # Fetch a larger candidate pool, then rank with BM25 (more recall for numeric queries)
            candidate_limit = max(300, k * 40)
            result = await session.execute(base_query.limit(candidate_limit))
            rows = result.all()
            if not rows:
                return []
            
            # Prepare documents for BM25
            docs_tokens: List[List[str]] = []
            chunks_meta: List[Tuple[ChunkModel, DocumentModel]] = []
            for chunk, document in rows:
                tokens = self._tokenize(chunk.content)
                if tokens:
                    docs_tokens.append(tokens)
                    chunks_meta.append((chunk, document))

            if not docs_tokens:
                return []

            # Compute BM25 scores
            scores = self._bm25_scores(docs_tokens, query_terms)

            # Build results: sort by score desc
            ranked = sorted(
                zip(chunks_meta, scores), key=lambda x: x[1], reverse=True
            )

            results: List[SearchResult] = []
            for (chunk, document), score in ranked[:k]:
                results.append(
                    SearchResult(
                    chunk_id=str(chunk.id),
                    document_id=str(chunk.document_id),
                    content=chunk.content,
                        score=float(score),
                    metadata=chunk.extra_metadata or {},
                        document_metadata=document.extra_metadata or {},
                    )
                )
                
            return results
            
    def _apply_filters(self, query, filters: Dict[str, Any]):
        """Apply filters to the query."""
        # Apply document-level filters
        if 'document_type' in filters:
            query = query.where(DocumentModel.extra_metadata['document_type'].astext == filters['document_type'])
            
        if 'date_range' in filters:
            start_date, end_date = filters['date_range']
            query = query.where(
                and_(DocumentModel.created_at >= start_date, DocumentModel.created_at <= end_date)
            )
            
        # Apply chunk-level metadata filters
        if 'chunk_metadata' in filters and isinstance(filters['chunk_metadata'], dict):
            for key, value in filters['chunk_metadata'].items():
                query = query.where(ChunkModel.extra_metadata[key].astext == str(value))

        return query

    # ---- BM25 utilities ----
    def _tokenize(self, text: str) -> List[str]:
        text = (text or "").lower()
        # Simple alphanumeric word tokenizer
        return re.findall(r"[a-z0-9]+", text)

    def _bm25_scores(
        self,
        docs_tokens: List[List[str]],
        query_terms: List[str],
        k1: float = None,
        b: float = None,
    ) -> List[float]:
        """Compute BM25 scores for each document in docs_tokens given query_terms."""
        from collections import Counter, defaultdict
        
        # Use config values if not provided
        if k1 is None:
            k1 = settings.bm25_k1
        if b is None:
            b = settings.bm25_b

        N = len(docs_tokens)
        # Document lengths and average length
        doc_lengths = [len(doc) for doc in docs_tokens]
        avgdl = sum(doc_lengths) / max(1, N)

        # Document frequency per term
        df = defaultdict(int)
        for doc in docs_tokens:
            seen = set(doc)
            for t in query_terms:
                if t in seen:
                    df[t] += 1

        # IDF per term (BM25 idf with added 1 per Robertson/Sparck-Jones variant)
        import math
        idf = {}
        for t in query_terms:
            n_q = df.get(t, 0)
            idf[t] = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1)

        scores: List[float] = []
        for doc_idx, doc in enumerate(docs_tokens):
            freq = Counter(doc)
            dl = doc_lengths[doc_idx]
            score = 0.0
            for t in query_terms:
                if t not in freq:
                    continue
                tf = freq[t]
                denom = tf + k1 * (1 - b + b * (dl / max(1.0, avgdl)))
                score += idf[t] * ((tf * (k1 + 1)) / denom)
            scores.append(score)
        return scores


class SemanticSearcher:
    """Semantic search using vector embeddings."""
    
    def __init__(self, embedding_service: EmbeddingService, vector_store: VectorStore):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        
    async def search(
        self, 
        query: str, 
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform semantic search."""
        # Use vector store to find similar chunks
        logger.info(f"Semantic search: query='{query}', k={k}")
        # Expand candidate pool for better recall on CPU-only setups (accept higher latency)
        vector_results = await self.vector_store.search(query, k=k*3, filters=filters)
        logger.info(f"Semantic search: raw_candidates={len(vector_results)}")

        # Apply similarity threshold to filter weak matches with safe fallback
        threshold = getattr(settings, 'similarity_threshold', 0.0) or 0.0
        if threshold > 0:
            filtered = [r for r in vector_results if r.get('score', 0) >= threshold]
            # If all results are filtered out, fall back to the original top results
            if filtered:
                vector_results = filtered
        logger.info(f"Semantic search: candidates_after_threshold={len(vector_results)} (threshold={threshold})")
        
        # Convert to SearchResult objects
        search_results = []
        
        async with get_db_session() as session:
            for result in vector_results[:k]:
                # Get chunk and document details
                chunk_result = await session.execute(
                    select(ChunkModel, DocumentModel)
                    .join(DocumentModel, ChunkModel.document_id == DocumentModel.id)
                    .where(ChunkModel.id == result['chunk_id'])
                )
                chunk_row = chunk_result.first()
                
                if chunk_row:
                    chunk, document = chunk_row
                    search_result = SearchResult(
                        chunk_id=str(chunk.id),
                        document_id=str(chunk.document_id),
                        content=chunk.content,
                        score=result['score'],
                        metadata=chunk.extra_metadata or {},
                        document_metadata=document.extra_metadata or {}
                    )
                    search_results.append(search_result)
                else:
                    logger.warning(f"Semantic search: chunk not found for id={result.get('chunk_id')}")
                    # Fallback: use vector-returned document text when DB chunk missing
                    fallback_text = result.get('document')
                    if fallback_text:
                        metadata = result.get('metadata') or {}
                        document_id = None
                        if isinstance(metadata, dict):
                            document_id = metadata.get('document_id')
                        search_result = SearchResult(
                            chunk_id=str(result.get('chunk_id') or result.get('raw_id') or ''),
                            document_id=str(document_id) if document_id else '',
                            content=fallback_text,
                            score=result.get('score', 0),
                            metadata=metadata if isinstance(metadata, dict) else {},
                            document_metadata={}
                        )
                        search_results.append(search_result)
                    
        return search_results


class HybridRetriever:
    """Hybrid retrieval combining semantic and keyword search."""
    
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.vector_store = VectorStore(embedding_service)
        self.keyword_searcher = KeywordSearcher()
        self.semantic_searcher = SemanticSearcher(embedding_service, self.vector_store)
        self.query_processor = QueryProcessor()
        self._cross_encoder: Optional[CrossEncoder] = None
        
    async def retrieve(
        self, 
        query: str, 
        k: int = 5,
        search_type: SearchType = SearchType.HYBRID,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Retrieve relevant documents using hybrid approach."""
        
        # Process query
        processed_query = await self.query_processor.process_query(query)
        
        # Check cache first
        cache_key = self._get_cache_key(query, k, search_type, filters)
        cached_results = await cache.get(cache_key)
        if cached_results:
            logger.info(f"Cache hit for query: {query[:50]}...")
            return cached_results
            
        if search_type == SearchType.SEMANTIC:
            results = await self._semantic_search(processed_query, k, filters)
        elif search_type == SearchType.KEYWORD:
            results = await self._keyword_search(processed_query, k, filters)
        else:  # HYBRID
            results = await self._hybrid_search(processed_query, k, filters)
            
        # Cache results
        await cache.set(cache_key, results, ttl=settings.query_cache_ttl)
        
        return results
        
    async def _semantic_search(
        self, 
        processed_query: Dict[str, Any], 
        k: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """Perform semantic search."""
        return await self.semantic_searcher.search(
            processed_query['original_query'], k=k, filters=filters
        )
        
    async def _keyword_search(
        self, 
        processed_query: Dict[str, Any], 
        k: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """Perform keyword search."""
        return await self.keyword_searcher.search(
            processed_query['processed_query'], 
            k=k, 
            filters=filters
        )
        
    async def _hybrid_search(
        self, 
        processed_query: Dict[str, Any], 
        k: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """Perform hybrid search combining semantic and keyword results."""
        
        # Run both searches in parallel
        semantic_task = self.semantic_searcher.search(
            processed_query['original_query'], k=k*3, filters=filters
        )
        keyword_task = self.keyword_searcher.search(
            processed_query['processed_query'], 
            k=k*3, 
            filters=filters
        )
        
        semantic_results, keyword_results = await asyncio.gather(
            semantic_task, keyword_task
        )
        logger.info(f"Hybrid retrieval: semantic={len(semantic_results)}, keyword={len(keyword_results)}")
        
        # If one side returns empty, fall back to the other
        if not semantic_results and keyword_results:
            return self._mmr_select(keyword_results, k)
        if not keyword_results and semantic_results:
            return self._mmr_select(semantic_results, k)

        # Combine and rerank results
        combined_results = self._combine_results(
            semantic_results, 
            keyword_results, 
            # Favor semantic relevance more for English-only corpora
            semantic_weight=0.8,
            keyword_weight=0.2
        )
        
        # Apply reciprocal rank fusion
        fused_results = self._reciprocal_rank_fusion([semantic_results, keyword_results])

        # Optional cross-encoder reranking over top-N
        reranked = await self._maybe_rerank(processed_query['original_query'], fused_results)

        # Apply MMR diversification; for numeric/count queries, bias toward relevance
        is_numeric_q = bool(re.search(r"\b(how many|how much|number of|count of)\b", processed_query['original_query'].lower()))
        lambda_balance = 0.9 if is_numeric_q else None
        diversified = self._mmr_select(reranked, k, lambda_balance=lambda_balance)
        return diversified

    async def _maybe_rerank(self, query: str, candidates: List[SearchResult]) -> List[SearchResult]:
        """Optionally rerank with a cross-encoder over top-N candidates."""
        if not getattr(settings, 'use_cross_encoder', True) or not candidates:
            return candidates
        try:
            top_n = min(len(candidates), getattr(settings, 'rerank_top_k', 10))
            subset = candidates[:top_n]
            # Lazy-load cross-encoder
            if self._cross_encoder is None:
                self._cross_encoder = CrossEncoder(settings.cross_encoder_model)
            # Prepare pairs (query, passage)
            pairs = [(query, c.content) for c in subset]
            import asyncio
            loop = asyncio.get_event_loop()
            scores = await loop.run_in_executor(None, self._cross_encoder.predict, pairs)
            # Assign scores and sort
            for i, s in enumerate(scores):
                subset[i].score = float(s)
            # Keep reranked subset first, then append remaining
            reranked_subset = sorted(subset, key=lambda x: x.score, reverse=True)
            remaining = candidates[top_n:]
            return reranked_subset + remaining
        except Exception as e:
            logger.warning(f"Cross-encoder rerank failed or unavailable: {e}")
            return candidates
        
    def _combine_results(
        self, 
        semantic_results: List[SearchResult],
        keyword_results: List[SearchResult],
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[SearchResult]:
        """Combine results from different search methods."""
        
        # Create a map of chunk_id to results
        result_map = {}
        
        # Add semantic results
        for result in semantic_results:
            result_map[result.chunk_id] = SearchResult(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                content=result.content,
                score=result.score * semantic_weight,
                metadata=result.metadata,
                document_metadata=result.document_metadata
            )
            
        # Add or combine keyword results
        for result in keyword_results:
            if result.chunk_id in result_map:
                # Combine scores
                existing = result_map[result.chunk_id]
                existing.score += result.score * keyword_weight
            else:
                result_map[result.chunk_id] = SearchResult(
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    content=result.content,
                    score=result.score * keyword_weight,
                    metadata=result.metadata,
                    document_metadata=result.document_metadata
                )
                
        # Sort by combined score
        combined_results = list(result_map.values())
        combined_results.sort(key=lambda x: x.score, reverse=True)
        
        return combined_results
        
    def _reciprocal_rank_fusion(
        self, 
        result_lists: List[List[SearchResult]], 
        k: int = 60
    ) -> List[SearchResult]:
        """Apply reciprocal rank fusion to combine multiple result lists."""
        
        # Calculate RRF scores
        rrf_scores = {}
        
        for result_list in result_lists:
            for rank, result in enumerate(result_list, 1):
                if result.chunk_id not in rrf_scores:
                    rrf_scores[result.chunk_id] = {
                        'score': 0.0,
                        'result': result
                    }
                rrf_scores[result.chunk_id]['score'] += 1.0 / (k + rank)
                
        # Sort by RRF score
        sorted_results = sorted(
            rrf_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        # Update scores and return results
        fused_results = []
        for item in sorted_results:
            result = item['result']
            result.score = item['score']
            fused_results.append(result)
            
        return fused_results

    def _mmr_select(
        self,
        candidates: List[SearchResult],
        k: int,
        lambda_balance: float = None
    ) -> List[SearchResult]:
        """Select top-k results with Maximal Marginal Relevance diversification.

        Requires a similarity function; we approximate with content overlap Jaccard over tokens.
        """
        if lambda_balance is None:
            lambda_balance = settings.mmr_lambda_balance
            
        if not candidates:
            return []
        k = max(1, min(k, len(candidates)))

        # Pre-tokenize candidates for simple Jaccard similarity
        def tokenize(text: str) -> set:
            return set(re.findall(r"[a-z0-9]+", (text or "").lower()))

        tokens = [tokenize(c.content) for c in candidates]
        selected: List[int] = []
        remaining = set(range(len(candidates)))

        # Normalize scores between 0 and 1
        scores = [c.score for c in candidates]
        if scores:
            min_s, max_s = min(scores), max(scores)
            if max_s > min_s:
                norm_scores = [(s - min_s) / (max_s - min_s) for s in scores]
            else:
                norm_scores = [0.0 for _ in scores]
        else:
            norm_scores = [0.0 for _ in candidates]

        def jaccard(a: set, b: set) -> float:
            if not a or not b:
                return 0.0
            inter = len(a & b)
            union = len(a | b)
            return inter / union if union else 0.0

        while len(selected) < k and remaining:
            best_idx = None
            best_score = -1.0
            for i in list(remaining):
                # Relevance component
                rel = norm_scores[i]
                # Diversity penalty: max similarity to any already selected
                if not selected:
                    sim = 0.0
                else:
                    sim = max(jaccard(tokens[i], tokens[j]) for j in selected)
                mmr = lambda_balance * rel - (1 - lambda_balance) * sim
                if mmr > best_score:
                    best_score = mmr
                    best_idx = i
            selected.append(best_idx)
            remaining.remove(best_idx)

        return [candidates[i] for i in selected]
        
    def _get_cache_key(
        self, 
        query: str, 
        k: int, 
        search_type: SearchType,
        filters: Optional[Dict[str, Any]]
    ) -> str:
        """Generate cache key for retrieval results."""
        import hashlib
        
        key_parts = [
            query.lower().strip(),
            str(k),
            search_type.value,
            str(sorted(filters.items()) if filters else "")
        ]
        
        key_string = "|".join(key_parts)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()
        
        return f"retrieval:v1:{key_hash}"
        
    async def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        return {
            "total_queries": 0,  # Placeholder
            "cache_hit_rate": 0.0,  # Placeholder
            "average_response_time": 0.0,  # Placeholder
            "search_types_used": {
                "semantic": 0,
                "keyword": 0,
                "hybrid": 0
            }
        }
