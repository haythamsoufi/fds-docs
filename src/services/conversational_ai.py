"""Advanced conversational AI service with memory, intent classification, and response quality improvements."""

import asyncio
import json
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import logging

from src.core.config import settings
from src.core.cache import cache
from src.core.monitoring import monitoring_service
from src.core.database import get_db_session
from src.core.models import QueryModel, QueryIntent
from sqlalchemy import select, func

logger = logging.getLogger(__name__)


@dataclass
class ConversationMessage:
    """Represents a message in a conversation."""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationContext:
    """Maintains context across conversation turns."""
    conversation_id: str
    messages: List[ConversationMessage] = field(default_factory=list)
    summary: Optional[str] = None
    last_activity: datetime = field(default_factory=datetime.utcnow)
    user_intent: Optional[QueryIntent] = None
    entities: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)

    def add_message(self, message: ConversationMessage):
        """Add a message to the conversation."""
        self.messages.append(message)
        self.last_activity = datetime.utcnow()

        # Keep only recent messages (last 20)
        if len(self.messages) > 20:
            self.messages = self.messages[-20:]

    def get_recent_messages(self, limit: int = 10) -> List[ConversationMessage]:
        """Get recent messages."""
        return self.messages[-limit:] if self.messages else []

    def get_context_summary(self) -> str:
        """Generate a summary of the conversation context."""
        if not self.messages:
            return "No conversation history"

        # Extract key topics and entities from recent messages
        recent_content = " ".join([msg.content for msg in self.messages[-5:]])

        # Simple topic extraction (in production, use NLP)
        topics = []
        if "document" in recent_content.lower():
            topics.append("documents")
        if "search" in recent_content.lower() or "query" in recent_content.lower():
            topics.append("search")
        if "upload" in recent_content.lower():
            topics.append("file upload")

        context = f"Conversation about: {', '.join(topics) if topics else 'general topics'}"
        if self.user_intent:
            context += f" | Intent: {self.user_intent.value}"

        return context


class IntentClassifier:
    """Advanced intent classification using patterns and context."""

    def __init__(self):
        self.intent_patterns = {
            QueryIntent.FACTUAL: [
                r'\b(what|who|when|where|which|how many|how much)\b.*\?',
                r'\b(tell me about|explain|describe|define)\b',
                r'\b(information|details|facts)\b',
            ],
            QueryIntent.COMPARISON: [
                r'\b(compare|difference|versus|vs|better|worse|similar|different)\b',
                r'\b(advantages?|disadvantages?|pros?|cons?)\b',
            ],
            QueryIntent.ANALYTICAL: [
                r'\b(analyze|evaluate|assess|examine|impact|effect|why|how does)\b',
                r'\b(cause|reason|explanation|analysis)\b',
            ],
            QueryIntent.PROCEDURAL: [
                r'\b(how to|steps?|process|procedure|instructions?|guide)\b',
                r'\b(create|build|make|implement|setup)\b',
            ]
        }

        self.intent_keywords = {
            QueryIntent.FACTUAL: ['what', 'who', 'when', 'where', 'which', 'tell', 'explain', 'information'],
            QueryIntent.COMPARISON: ['compare', 'difference', 'versus', 'vs', 'better', 'worse', 'similar'],
            QueryIntent.ANALYTICAL: ['analyze', 'evaluate', 'why', 'how does', 'impact', 'effect', 'cause'],
            QueryIntent.PROCEDURAL: ['how to', 'steps', 'process', 'create', 'build', 'make', 'guide']
        }

    async def classify_intent(self, query: str, context: Optional[ConversationContext] = None) -> Tuple[QueryIntent, float]:
        """Classify query intent with confidence score."""
        query_lower = query.lower()
        scores = {intent: 0.0 for intent in QueryIntent}

        # Pattern matching
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    scores[intent] += 1.0

        # Keyword matching
        for intent, keywords in self.intent_keywords.items():
            keyword_matches = sum(1 for keyword in keywords if keyword in query_lower)
            scores[intent] += keyword_matches * 0.5

        # Context influence
        if context and context.user_intent:
            # Boost score for same intent as previous messages
            scores[context.user_intent] += 0.3

        # Find best intent
        best_intent = max(scores.items(), key=lambda x: x[1])
        confidence = best_intent[1] / max(sum(scores.values()), 1.0)

        # Return factual if confidence is too low
        if confidence < 0.3:
            return QueryIntent.FACTUAL, confidence

        return best_intent[0], confidence


class ResponseQualityEnhancer:
    """Enhances response quality with citations, formatting, and clarity."""

    def __init__(self):
        self.citation_patterns = [
            r'source\s+(\d+)',
            r'\[(\d+)\]',
            r'\((\d+)\)',
        ]

    async def enhance_response(self, query: str, raw_response: str,
                              retrieved_chunks: List, context: Optional[ConversationContext] = None) -> Dict[str, Any]:
        """Enhance response with citations, formatting, and quality improvements."""
        enhanced = {
            "original_response": raw_response,
            "enhanced_response": raw_response,
            "citations": [],
            "confidence": 0.8,
            "clarity_score": 0.7,
            "completeness_score": 0.8,
            "formatting": {}
        }

        # Add citations
        enhanced["citations"] = self._extract_citations(raw_response, retrieved_chunks)

        # Improve formatting
        enhanced["enhanced_response"] = self._improve_formatting(raw_response)

        # Add context awareness
        if context:
            enhanced["context_aware"] = self._add_context_awareness(raw_response, context)
            enhanced["enhanced_response"] = enhanced["context_aware"]["response"]

        # Calculate quality scores
        enhanced.update(self._calculate_quality_scores(query, raw_response, retrieved_chunks))

        return enhanced

    def _extract_citations(self, response: str, chunks: List) -> List[Dict[str, Any]]:
        """Extract and format citations."""
        citations = []

        for i, chunk in enumerate(chunks[:3]):  # Limit to top 3
            citation = {
                "index": i + 1,
                "chunk_id": getattr(chunk, 'chunk_id', str(chunk.id) if hasattr(chunk, 'id') else str(i)),
                "document_title": getattr(chunk, 'document', {}).get('filename', f'Document {i+1}') if hasattr(chunk, 'document') else f'Document {i+1}',
                "relevance_score": getattr(chunk, 'score', 0.5),
                "excerpt": self._get_relevant_excerpt(response, chunk)
            }
            citations.append(citation)

        return citations

    def _get_relevant_excerpt(self, response: str, chunk: Any) -> str:
        """Extract relevant excerpt from chunk."""
        chunk_content = getattr(chunk, 'content', '')

        # Find sentences in response that might relate to this chunk
        response_sentences = re.split(r'[.!?]+', response)
        chunk_sentences = re.split(r'[.!?]+', chunk_content)

        # Simple matching (in production, use more sophisticated similarity)
        common_words = set(response.lower().split()) & set(chunk_content.lower().split())
        relevance = len(common_words) / max(len(set(response.lower().split())), 1)

        if relevance > 0.1:
            # Return first sentence of chunk as excerpt
            first_sentence = chunk_sentences[0].strip() if chunk_sentences else chunk_content[:100]
            return first_sentence + "..." if len(first_sentence) > 100 else first_sentence

        return chunk_content[:150] + "..." if len(chunk_content) > 150 else chunk_content

    def _improve_formatting(self, response: str) -> str:
        """Improve response formatting."""
        improved = response.strip()

        # Add line breaks for better readability
        improved = re.sub(r'(\.)\s+', r'\1\n\n', improved)

        # Ensure proper capitalization
        sentences = re.split(r'([.!?]+)', improved)
        capitalized_sentences = []
        for i in range(0, len(sentences), 2):
            sentence = sentences[i].strip()
            if sentence and not sentence[0].isupper() and len(sentence) > 3:
                sentence = sentence[0].upper() + sentence[1:]
            capitalized_sentences.append(sentence)
            if i + 1 < len(sentences):
                capitalized_sentences.append(sentences[i + 1])

        return ''.join(capitalized_sentences)

    def _add_context_awareness(self, response: str, context: ConversationContext) -> Dict[str, Any]:
        """Add context awareness to response."""
        enhanced_response = response

        # Add conversation context if this is a follow-up
        if len(context.messages) > 2:
            context_summary = context.get_context_summary()
            if context_summary != "No conversation history":
                enhanced_response = f"Based on our previous discussion about {context_summary}:\n\n{enhanced_response}"

        # Add intent-specific enhancements
        if context.user_intent == QueryIntent.PROCEDURAL:
            enhanced_response = self._enhance_procedural_response(enhanced_response)
        elif context.user_intent == QueryIntent.COMPARISON:
            enhanced_response = self._enhance_comparison_response(enhanced_response)

        return {
            "response": enhanced_response,
            "context_used": context_summary if 'context_summary' in locals() else None
        }

    def _enhance_procedural_response(self, response: str) -> str:
        """Enhance responses for procedural queries."""
        # Add step numbering if not present
        if "step" not in response.lower() and ("how to" in response.lower() or any(word in response.lower() for word in ["first", "then", "next", "after"])):
            lines = response.split('\n')
            numbered_lines = []
            step_num = 1

            for line in lines:
                stripped = line.strip()
                if (stripped and len(stripped) > 10 and
                    not stripped[0].isdigit() and
                    any(word in stripped.lower() for word in ["first", "then", "next", "after", "start", "begin"])):
                    numbered_lines.append(f"{step_num}. {stripped}")
                    step_num += 1
                else:
                    numbered_lines.append(line)

            return '\n'.join(numbered_lines)

        return response

    def _enhance_comparison_response(self, response: str) -> str:
        """Enhance responses for comparison queries."""
        # Add comparison table formatting if comparing items
        if any(word in response.lower() for word in ["better", "worse", "compare", "versus", "vs"]):
            # Try to extract comparison points
            sentences = re.split(r'[.!?]+', response)
            comparison_points = []

            for sentence in sentences:
                if any(word in sentence.lower() for word in ["better", "worse", "advantage", "disadvantage", "pros", "cons"]):
                    comparison_points.append(sentence.strip())

            if len(comparison_points) >= 2:
                # Add a simple comparison summary
                comparison_summary = "\n\n**Comparison Summary:**\n" + "\n".join(f"- {point}" for point in comparison_points[:3])
                return response + comparison_summary

        return response

    def _calculate_quality_scores(self, query: str, response: str, chunks: List) -> Dict[str, float]:
        """Calculate response quality scores."""
        scores = {
            "confidence": 0.8,
            "clarity_score": 0.7,
            "completeness_score": 0.8
        }

        # Confidence based on chunk relevance and response length
        if chunks:
            avg_score = sum(getattr(chunk, 'score', 0.5) for chunk in chunks) / len(chunks)
            scores["confidence"] = min(0.95, avg_score + 0.1)

        # Clarity based on response structure
        sentences = len(re.split(r'[.!?]+', response))
        words = len(response.split())

        if 5 <= sentences <= 20 and 50 <= words <= 500:
            scores["clarity_score"] = 0.9
        elif sentences < 3 or words < 20:
            scores["clarity_score"] = 0.5
        else:
            scores["clarity_score"] = 0.7

        # Completeness based on query coverage
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        coverage = len(query_words & response_words) / max(len(query_words), 1)

        if coverage > 0.6:
            scores["completeness_score"] = 0.9
        elif coverage > 0.3:
            scores["completeness_score"] = 0.7
        else:
            scores["completeness_score"] = 0.5

        return scores


class ConversationalMemory:
    """Manages conversation memory and context."""

    def __init__(self):
        self.active_conversations: Dict[str, ConversationContext] = {}
        self.max_conversations = 100

    def get_or_create_conversation(self, conversation_id: str) -> ConversationContext:
        """Get existing conversation or create new one."""
        if conversation_id not in self.active_conversations:
            self.active_conversations[conversation_id] = ConversationContext(
                conversation_id=conversation_id
            )

            # Cleanup old conversations if too many
            if len(self.active_conversations) > self.max_conversations:
                self._cleanup_old_conversations()

        return self.active_conversations[conversation_id]

    def _cleanup_old_conversations(self):
        """Remove old inactive conversations."""
        current_time = datetime.utcnow()
        cutoff_time = current_time - timedelta(hours=2)  # 2 hours inactive

        to_remove = []
        for conv_id, context in self.active_conversations.items():
            if context.last_activity < cutoff_time:
                to_remove.append(conv_id)

        for conv_id in to_remove:
            del self.active_conversations[conv_id]

        logger.info(f"Cleaned up {len(to_remove)} inactive conversations")

    async def store_query_response(self, conversation_id: str, query: str,
                                  response: str, intent: QueryIntent,
                                  entities: List[str] = None) -> ConversationContext:
        """Store a query-response pair in conversation memory."""
        context = self.get_or_create_conversation(conversation_id)

        # Add user message
        user_message = ConversationMessage(
            role="user",
            content=query,
            metadata={"intent": intent.value, "entities": entities or []}
        )
        context.add_message(user_message)

        # Add assistant message
        assistant_message = ConversationMessage(
            role="assistant",
            content=response,
            metadata={"intent": intent.value}
        )
        context.add_message(assistant_message)

        # Update context metadata
        context.user_intent = intent
        context.entities = entities or []

        # Update topics based on content
        context.topics = self._extract_topics(query + " " + response)

        return context

    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text (simplified)."""
        topics = []
        text_lower = text.lower()

        topic_keywords = {
            "documents": ["document", "file", "pdf", "upload"],
            "search": ["search", "query", "find", "lookup"],
            "analysis": ["analyze", "analysis", "evaluate", "assessment"],
            "technical": ["api", "system", "server", "database", "code"],
            "business": ["strategy", "plan", "process", "workflow", "management"]
        }

        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)

        return list(set(topics))[:3]  # Limit to 3 topics


class ConversationalAIService:
    """Advanced conversational AI service."""

    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.quality_enhancer = ResponseQualityEnhancer()
        self.memory = ConversationalMemory()

    async def process_query(self, query: str, conversation_id: str = "default") -> Dict[str, Any]:
        """Process a query with conversational context."""
        start_time = time.time()

        try:
            # Get conversation context
            context = self.memory.get_or_create_conversation(conversation_id)

            # Classify intent
            intent, confidence = await self.intent_classifier.classify_intent(query, context)

            # Extract entities (simplified)
            entities = self._extract_entities(query)

            # Update context
            context.user_intent = intent
            context.entities = entities

            # Track metrics
            await monitoring_service.log_event(
                "conversational_query",
                {
                    "conversation_id": conversation_id,
                    "intent": intent.value,
                    "confidence": confidence,
                    "entities_count": len(entities)
                }
            )

            processing_time = time.time() - start_time

            return {
                "query": query,
                "intent": intent,
                "confidence": confidence,
                "entities": entities,
                "context": context.get_context_summary(),
                "conversation_id": conversation_id,
                "processing_time": processing_time,
                "requires_llm": self._should_use_llm(query, intent)
            }

        except Exception as e:
            await monitoring_service.log_event(
                "conversational_query_failed",
                {"error": str(e), "conversation_id": conversation_id},
                level="error"
            )
            raise

    async def enhance_response(self, query: str, raw_response: str,
                              retrieved_chunks: List, conversation_id: str = "default") -> Dict[str, Any]:
        """Enhance response with conversational context."""
        # Get conversation context
        context = self.memory.get_or_create_conversation(conversation_id)

        # Enhance response
        enhanced = await self.quality_enhancer.enhance_response(
            query, raw_response, retrieved_chunks, context
        )

        # Store in conversation memory
        await self.memory.store_query_response(
            conversation_id, query, enhanced["enhanced_response"],
            context.user_intent or QueryIntent.FACTUAL
        )

        return enhanced

    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from query."""
        entities = []

        # Extract capitalized words (potential proper nouns)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', query)
        entities.extend(capitalized_words)

        # Extract numbers and dates
        numbers = re.findall(r'\b\d{4}\b|\b\d+\b', query)
        entities.extend(numbers)

        # Extract quoted phrases
        quoted = re.findall(r'"([^"]*)"', query)
        entities.extend(quoted)

        return list(set(entities))[:10]  # Limit entities

    def _should_use_llm(self, query: str, intent: QueryIntent) -> bool:
        """Determine if LLM should be used for this query."""
        # Use LLM for complex intents or long queries
        if intent in [QueryIntent.ANALYTICAL, QueryIntent.COMPARISON]:
            return True

        if len(query.split()) > 20:  # Long queries
            return True

        # Use extractive approach for simple factual queries
        if intent == QueryIntent.FACTUAL and len(query.split()) < 10:
            return False

        return True

    async def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        return {
            "active_conversations": len(self.memory.active_conversations),
            "total_messages": sum(len(ctx.messages) for ctx in self.memory.active_conversations.values()),
            "average_context_length": sum(len(ctx.messages) for ctx in self.memory.active_conversations.values()) / max(len(self.memory.active_conversations), 1)
        }


# Global conversational AI service instance
conversational_ai = ConversationalAIService()
