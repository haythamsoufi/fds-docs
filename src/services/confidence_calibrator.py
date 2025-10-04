"""Confidence calibration and no-answer threshold service."""

import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import math
import re

from src.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceScore:
    """Confidence score with calibration metadata."""
    raw_score: float
    calibrated_score: float
    is_no_answer: bool
    reasoning: str
    factors: Dict[str, float]


class ConfidenceCalibrator:
    """Calibrates confidence scores and determines no-answer thresholds."""

    def __init__(self):
        self.no_answer_threshold = settings.no_answer_threshold
        self.min_chunk_score = settings.min_chunk_score_threshold
        self.calibration_enabled = settings.confidence_calibration_enabled

    def calibrate_confidence(
        self,
        query: str,
        search_results: List,
        raw_answer: str,
        top_k_scores: List[float]
    ) -> ConfidenceScore:
        """Calibrate confidence score based on multiple factors."""
        
        if not self.calibration_enabled:
            # Simple fallback calibration
            raw_score = self._calculate_simple_confidence(top_k_scores)
            return ConfidenceScore(
                raw_score=raw_score,
                calibrated_score=raw_score,
                is_no_answer=raw_score < self.no_answer_threshold,
                reasoning="Simple calibration (disabled)",
                factors={"simple_avg": raw_score}
            )

        # Multi-factor calibration
        factors = self._calculate_confidence_factors(query, search_results, raw_answer, top_k_scores)
        
        # Weighted combination of factors
        weights = {
            "retrieval_quality": 0.4,
            "answer_coherence": 0.2,
            "source_coverage": 0.2,
            "query_match": 0.1,
            "response_length": 0.1
        }
        
        calibrated_score = sum(factors.get(factor, 0.0) * weight for factor, weight in weights.items())
        calibrated_score = max(0.0, min(1.0, calibrated_score))
        
        # Determine no-answer threshold
        # Allow brief numeric answers if retrieval quality is reasonable
        answer_text = (raw_answer or "").strip().lower()
        numeric_like = bool(re.search(r"\b(\d+[\d,\.]*)\b", answer_text))
        short_but_numeric = len(answer_text) < 20 and numeric_like and factors.get("retrieval_quality", 0.0) >= 0.2

        is_no_answer = (
            calibrated_score < (self.no_answer_threshold - 0.05 if short_but_numeric else self.no_answer_threshold) or
            len([s for s in top_k_scores if s >= self.min_chunk_score]) == 0 or
            (len(raw_answer.strip()) < 20 and not short_but_numeric)
        )
        
        reasoning = self._generate_reasoning(factors, is_no_answer)
        
        return ConfidenceScore(
            raw_score=factors.get("retrieval_quality", 0.0),
            calibrated_score=calibrated_score,
            is_no_answer=is_no_answer,
            reasoning=reasoning,
            factors=factors
        )

    def _calculate_confidence_factors(
        self,
        query: str,
        search_results: List,
        raw_answer: str,
        top_k_scores: List[float]
    ) -> Dict[str, float]:
        """Calculate individual confidence factors."""
        factors = {}
        
        # 1. Retrieval Quality (0-1)
        if top_k_scores:
            avg_score = sum(top_k_scores) / len(top_k_scores)
            max_score = max(top_k_scores)
            factors["retrieval_quality"] = (avg_score * 0.7 + max_score * 0.3)
        else:
            factors["retrieval_quality"] = 0.0
        
        # 2. Answer Coherence (0-1)
        factors["answer_coherence"] = self._assess_answer_coherence(raw_answer)
        
        # 3. Source Coverage (0-1)
        factors["source_coverage"] = self._assess_source_coverage(search_results, raw_answer)
        
        # 4. Query Match (0-1)
        factors["query_match"] = self._assess_query_match(query, raw_answer)
        
        # 5. Response Length Appropriateness (0-1)
        factors["response_length"] = self._assess_response_length(query, raw_answer)
        
        return factors

    def _calculate_simple_confidence(self, top_k_scores: List[float]) -> float:
        """Simple confidence calculation as fallback."""
        if not top_k_scores:
            return 0.0
        return min(1.0, max(0.0, sum(top_k_scores) / len(top_k_scores)))

    def _assess_answer_coherence(self, answer: str) -> float:
        """Assess answer coherence (0-1)."""
        if not answer or len(answer.strip()) < 5:
            return 0.0
        
        # Check for no-answer indicators
        no_answer_phrases = [
            "i don't know", "i cannot", "no information", "not found",
            "insufficient", "unclear", "cannot determine", "not available"
        ]
        
        answer_lower = answer.lower()
        for phrase in no_answer_phrases:
            if phrase in answer_lower:
                return 0.3  # Low but not zero for explicit no-answer
        
        # Check for good answer indicators
        good_indicators = ["according to", "based on", "the sources", "specifically"]
        good_score = sum(1 for indicator in good_indicators if indicator in answer_lower)
        
        # Length appropriateness (not too short, not too long). Allow short numeric replies.
        length_score = 1.0
        if len(answer) < 30:
            if re.search(r"\b(\d+[\d,\.]*)\b", answer.lower()):
                length_score = 0.8
            else:
                length_score = 0.3
        elif len(answer) > 1000:
            length_score = 0.8
        
        return min(1.0, (good_score * 0.3 + length_score * 0.7))

    def _assess_source_coverage(self, search_results: List, answer: str) -> float:
        """Assess how well the answer covers the sources (0-1)."""
        if not search_results:
            return 0.0
        
        # Count citations in answer
        import re
        citations = re.findall(r'\[Source \d+\]', answer)
        citation_count = len(citations)
        
        # Coverage based on citations and result count
        if len(search_results) == 0:
            return 0.0
        elif citation_count == 0:
            return 0.2  # No citations
        else:
            return min(1.0, citation_count / min(3, len(search_results)))

    def _assess_query_match(self, query: str, answer: str) -> float:
        """Assess how well the answer matches the query (0-1)."""
        if not query or not answer:
            return 0.0
        
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        # Remove common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        query_words = query_words - stop_words
        answer_words = answer_words - stop_words
        
        if not query_words:
            return 0.5
        
        # Calculate overlap
        overlap = len(query_words & answer_words)
        coverage = overlap / len(query_words)
        
        return min(1.0, coverage)

    def _assess_response_length(self, query: str, answer: str) -> float:
        """Assess response length appropriateness (0-1)."""
        if not answer:
            return 0.0
        
        query_length = len(query.split())
        answer_length = len(answer.split())
        
        # Ideal length varies with query complexity
        if query_length <= 3:
            ideal_length = (20, 100)  # Simple questions
        elif query_length <= 8:
            ideal_length = (30, 200)  # Medium questions
        else:
            ideal_length = (50, 400)  # Complex questions
        
        min_length, max_length = ideal_length
        
        if answer_length < min_length:
            return 0.3  # Too short
        elif answer_length > max_length:
            return 0.7  # Too long but still useful
        else:
            return 1.0  # Appropriate length

    def _generate_reasoning(self, factors: Dict[str, float], is_no_answer: bool) -> str:
        """Generate human-readable reasoning for the confidence score."""
        if not self.calibration_enabled:
            return "Simple confidence calculation (calibration disabled)"
        
        reasons = []
        
        if is_no_answer:
            reasons.append("Below no-answer threshold")
        
        # Add factor-based reasoning
        low_factors = [name for name, score in factors.items() if score < 0.4]
        high_factors = [name for name, score in factors.items() if score > 0.7]
        
        if low_factors:
            reasons.append(f"Low {', '.join(low_factors)}")
        if high_factors:
            reasons.append(f"Strong {', '.join(high_factors)}")
        
        return "; ".join(reasons) if reasons else "Balanced confidence factors"

    def should_refuse_answer(self, confidence_score: ConfidenceScore) -> bool:
        """Determine if the system should refuse to answer."""
        return (
            confidence_score.is_no_answer or
            confidence_score.calibrated_score < self.no_answer_threshold or
            len(confidence_score.reasoning) == 0
        )

    def get_no_answer_response(self, query: str) -> str:
        """Generate appropriate no-answer response."""
        return (
            "I don't have sufficient information in the available sources to answer "
            "your question about '{}'. Please try rephrasing your query or ask about "
            "a different aspect of the topic."
        ).format(query)


# Global instance
confidence_calibrator = ConfidenceCalibrator()
