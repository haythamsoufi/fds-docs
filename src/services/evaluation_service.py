"""RAGAS evaluation service for RAG system quality assessment."""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import pandas as pd

from src.core.config import settings

logger = logging.getLogger(__name__)

# Type hints for when RAGAS is available
if TYPE_CHECKING:
    from datasets import Dataset

# Optional imports with fallbacks
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        context_utilization,
        response_completeness,
        response_consistency
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logger.warning("RAGAS not available. Install ragas and datasets for evaluation support.")


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    query: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None
    
    # RAGAS Metrics
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    context_utilization: Optional[float] = None
    response_completeness: Optional[float] = None
    response_consistency: Optional[float] = None
    
    # Custom Metrics
    response_time: Optional[float] = None
    context_count: Optional[int] = None
    answer_length: Optional[int] = None
    
    # Metadata
    timestamp: Optional[str] = None
    model_version: Optional[str] = None
    evaluation_version: Optional[str] = None


@dataclass
class EvaluationDataset:
    """Container for evaluation dataset."""
    name: str
    description: str
    queries: List[str]
    ground_truths: List[Optional[str]]
    expected_contexts: List[List[str]]
    metadata: Dict[str, Any]


class RAGASEvaluator:
    """RAGAS-based evaluation service for RAG system."""
    
    def __init__(self):
        self.ragas_available = RAGAS_AVAILABLE
        self.evaluation_version = "1.0.0"
        
    async def evaluate_query(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> EvaluationResult:
        """Evaluate a single query-answer pair."""
        
        result = EvaluationResult(
            query=query,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
            context_count=len(contexts),
            answer_length=len(answer),
            timestamp=datetime.now().isoformat(),
            model_version=getattr(settings, 'embedding_version', '1.0'),
            evaluation_version=self.evaluation_version
        )
        
        if not self.ragas_available:
            logger.warning("RAGAS not available, returning basic metrics only")
            return result
        
        try:
            # Prepare data for RAGAS evaluation
            eval_data = {
                "question": [query],
                "answer": [answer],
                "contexts": [contexts],
                "ground_truth": [ground_truth] if ground_truth else [None]
            }
            
            # Create dataset
            dataset = Dataset.from_dict(eval_data)
            
            # Define metrics
            metrics = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                response_completeness
            ]
            
            # Add ground truth metrics if available
            if ground_truth:
                metrics.extend([
                    context_utilization,
                    response_consistency
                ])
            
            # Run evaluation
            evaluation_result = await self._run_ragas_evaluation(dataset, metrics)
            
            # Extract scores
            if evaluation_result:
                result.faithfulness = evaluation_result.get('faithfulness', {}).get('score', 0.0)
                result.answer_relevancy = evaluation_result.get('answer_relevancy', {}).get('score', 0.0)
                result.context_precision = evaluation_result.get('context_precision', {}).get('score', 0.0)
                result.context_recall = evaluation_result.get('context_recall', {}).get('score', 0.0)
                result.response_completeness = evaluation_result.get('response_completeness', {}).get('score', 0.0)
                
                if ground_truth:
                    result.context_utilization = evaluation_result.get('context_utilization', {}).get('score', 0.0)
                    result.response_consistency = evaluation_result.get('response_consistency', {}).get('score', 0.0)
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            
        return result
    
    async def evaluate_batch(
        self,
        queries: List[str],
        answers: List[str],
        contexts_list: List[List[str]],
        ground_truths: Optional[List[Optional[str]]] = None
    ) -> List[EvaluationResult]:
        """Evaluate a batch of query-answer pairs."""
        
        if not self.ragas_available:
            logger.warning("RAGAS not available, returning basic metrics only")
            return [
                EvaluationResult(
                    query=q,
                    answer=a,
                    contexts=c,
                    ground_truth=g,
                    context_count=len(c),
                    answer_length=len(a),
                    timestamp=datetime.now().isoformat(),
                    model_version=getattr(settings, 'embedding_version', '1.0'),
                    evaluation_version=self.evaluation_version
                )
                for q, a, c, g in zip(queries, answers, contexts_list, ground_truths or [None] * len(queries))
            ]
        
        try:
            # Prepare batch data
            eval_data = {
                "question": queries,
                "answer": answers,
                "contexts": contexts_list,
                "ground_truth": ground_truths or [None] * len(queries)
            }
            
            # Create dataset
            dataset = Dataset.from_dict(eval_data)
            
            # Define metrics
            metrics = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                response_completeness
            ]
            
            # Add ground truth metrics if available
            if ground_truths and any(gt for gt in ground_truths):
                metrics.extend([
                    context_utilization,
                    response_consistency
                ])
            
            # Run evaluation
            evaluation_result = await self._run_ragas_evaluation(dataset, metrics)
            
            # Convert to EvaluationResult objects
            results = []
            for i, (query, answer, contexts, ground_truth) in enumerate(zip(queries, answers, contexts_list, ground_truths or [None] * len(queries))):
                result = EvaluationResult(
                    query=query,
                    answer=answer,
                    contexts=contexts,
                    ground_truth=ground_truth,
                    context_count=len(contexts),
                    answer_length=len(answer),
                    timestamp=datetime.now().isoformat(),
                    model_version=getattr(settings, 'embedding_version', '1.0'),
                    evaluation_version=self.evaluation_version
                )
                
                # Extract scores for this item
                if evaluation_result:
                    result.faithfulness = evaluation_result.get('faithfulness', {}).get('score', 0.0)
                    result.answer_relevancy = evaluation_result.get('answer_relevancy', {}).get('score', 0.0)
                    result.context_precision = evaluation_result.get('context_precision', {}).get('score', 0.0)
                    result.context_recall = evaluation_result.get('context_recall', {}).get('score', 0.0)
                    result.response_completeness = evaluation_result.get('response_completeness', {}).get('score', 0.0)
                    
                    if ground_truth:
                        result.context_utilization = evaluation_result.get('context_utilization', {}).get('score', 0.0)
                        result.response_consistency = evaluation_result.get('response_consistency', {}).get('score', 0.0)
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch RAGAS evaluation failed: {e}")
            # Return basic results
            return [
                EvaluationResult(
                    query=q,
                    answer=a,
                    contexts=c,
                    ground_truth=g,
                    context_count=len(c),
                    answer_length=len(a),
                    timestamp=datetime.now().isoformat(),
                    model_version=getattr(settings, 'embedding_version', '1.0'),
                    evaluation_version=self.evaluation_version
                )
                for q, a, c, g in zip(queries, answers, contexts_list, ground_truths or [None] * len(queries))
            ]
    
    async def _run_ragas_evaluation(self, dataset: Any, metrics: List) -> Optional[Dict]:
        """Run RAGAS evaluation in async context."""
        try:
            # Run evaluation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: evaluate(dataset, metrics=metrics)
            )
            return result
        except Exception as e:
            logger.error(f"RAGAS evaluation execution failed: {e}")
            return None
    
    def calculate_overall_scores(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """Calculate overall evaluation scores."""
        if not results:
            return {}
        
        scores = {}
        
        # Calculate averages for each metric
        metrics = [
            'faithfulness', 'answer_relevancy', 'context_precision',
            'context_recall', 'response_completeness', 'context_utilization',
            'response_consistency'
        ]
        
        for metric in metrics:
            values = [getattr(r, metric) for r in results if getattr(r, metric) is not None]
            if values:
                scores[f"{metric}_avg"] = sum(values) / len(values)
                scores[f"{metric}_min"] = min(values)
                scores[f"{metric}_max"] = max(values)
        
        # Calculate custom metrics
        response_times = [r.response_time for r in results if r.response_time is not None]
        if response_times:
            scores["response_time_avg"] = sum(response_times) / len(response_times)
            scores["response_time_min"] = min(response_times)
            scores["response_time_max"] = max(response_times)
        
        context_counts = [r.context_count for r in results if r.context_count is not None]
        if context_counts:
            scores["context_count_avg"] = sum(context_counts) / len(context_counts)
        
        answer_lengths = [r.answer_length for r in results if r.answer_length is not None]
        if answer_lengths:
            scores["answer_length_avg"] = sum(answer_lengths) / len(answer_lengths)
        
        return scores
    
    def export_results(self, results: List[EvaluationResult], format: str = "json") -> str:
        """Export evaluation results to specified format."""
        if format == "json":
            data = [asdict(result) for result in results]
            return json.dumps(data, indent=2)
        elif format == "csv":
            df = pd.DataFrame([asdict(result) for result in results])
            return df.to_csv(index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_evaluation_status(self) -> Dict[str, Any]:
        """Get evaluation service status."""
        return {
            "ragas_available": self.ragas_available,
            "evaluation_version": self.evaluation_version,
            "supported_metrics": [
                "faithfulness", "answer_relevancy", "context_precision",
                "context_recall", "context_utilization", "response_completeness",
                "response_consistency"
            ] if self.ragas_available else [],
            "custom_metrics": [
                "response_time", "context_count", "answer_length"
            ]
        }


class EvaluationDatasetManager:
    """Manager for evaluation datasets."""
    
    def __init__(self):
        self.datasets = {}
        self._load_sample_datasets()
    
    def _load_sample_datasets(self):
        """Load sample evaluation datasets."""
        
        # Sample dataset 1: Company Policy Questions
        policy_dataset = EvaluationDataset(
            name="company_policy",
            description="Sample questions about company policies and procedures",
            queries=[
                "What is the company's remote work policy?",
                "How many vacation days do employees get per year?",
                "What is the procedure for requesting time off?",
                "What are the company's core values?",
                "How does the performance review process work?",
                "What is the dress code policy?",
                "What are the working hours?",
                "How do I report a workplace incident?",
                "What benefits does the company offer?",
                "What is the company's policy on professional development?"
            ],
            ground_truths=[
                "Employees can work remotely up to 3 days per week with manager approval",
                "Employees receive 20 vacation days per year, plus 10 sick days",
                "Submit time-off requests through the HR portal at least 2 weeks in advance",
                "Our core values are integrity, innovation, collaboration, and excellence",
                "Performance reviews are conducted annually with quarterly check-ins",
                "Business casual attire is required in the office",
                "Standard working hours are 9 AM to 5 PM, Monday through Friday",
                "Report incidents immediately to your supervisor and HR",
                "Health insurance, dental, vision, 401k matching, and gym membership",
                "The company provides up to $2,000 per year for professional development"
            ],
            expected_contexts=[
                ["Remote work policy allows up to 3 days per week"],
                ["Vacation policy includes 20 days annual leave"],
                ["Time-off requests require 2-week advance notice"],
                ["Company values include integrity and innovation"],
                ["Performance reviews are annual with quarterly check-ins"],
                ["Dress code is business casual"],
                ["Working hours are 9 AM to 5 PM"],
                ["Incident reporting procedures"],
                ["Benefits package includes health insurance"],
                ["Professional development budget is $2,000 annually"]
            ],
            metadata={
                "domain": "human_resources",
                "difficulty": "medium",
                "created_date": "2024-01-01"
            }
        )
        
        # Sample dataset 2: Technical Documentation
        tech_dataset = EvaluationDataset(
            name="technical_docs",
            description="Technical questions about system architecture and APIs",
            queries=[
                "How do I authenticate with the API?",
                "What is the rate limit for API requests?",
                "How do I handle pagination in API responses?",
                "What are the supported data formats?",
                "How do I implement error handling?",
                "What is the system architecture?",
                "How do I deploy the application?",
                "What monitoring tools are available?",
                "How do I configure logging?",
                "What are the security best practices?"
            ],
            ground_truths=[
                "Use API key authentication with Bearer token in Authorization header",
                "Rate limit is 1000 requests per hour per API key",
                "Use offset and limit parameters for pagination",
                "Supported formats are JSON, XML, and CSV",
                "Implement try-catch blocks and handle HTTP status codes",
                "Microservices architecture with containerized deployment",
                "Deploy using Docker containers and Kubernetes",
                "Prometheus and Grafana for monitoring",
                "Configure logging levels in application.properties",
                "Use HTTPS, input validation, and rate limiting"
            ],
            expected_contexts=[
                ["API authentication using Bearer tokens"],
                ["Rate limiting of 1000 requests per hour"],
                ["Pagination with offset and limit parameters"],
                ["JSON, XML, and CSV format support"],
                ["Error handling with HTTP status codes"],
                ["Microservices architecture design"],
                ["Docker and Kubernetes deployment"],
                ["Prometheus and Grafana monitoring"],
                ["Logging configuration options"],
                ["Security practices and recommendations"]
            ],
            metadata={
                "domain": "technical",
                "difficulty": "high",
                "created_date": "2024-01-01"
            }
        )
        
        self.datasets["company_policy"] = policy_dataset
        self.datasets["technical_docs"] = tech_dataset
    
    def get_dataset(self, name: str) -> Optional[EvaluationDataset]:
        """Get evaluation dataset by name."""
        return self.datasets.get(name)
    
    def list_datasets(self) -> List[str]:
        """List available dataset names."""
        return list(self.datasets.keys())
    
    def add_dataset(self, dataset: EvaluationDataset):
        """Add a new evaluation dataset."""
        self.datasets[dataset.name] = dataset
    
    def export_dataset(self, name: str, format: str = "json") -> Optional[str]:
        """Export dataset to specified format."""
        dataset = self.get_dataset(name)
        if not dataset:
            return None
        
        if format == "json":
            return json.dumps(asdict(dataset), indent=2)
        elif format == "csv":
            data = []
            for i, (query, gt, contexts) in enumerate(zip(dataset.queries, dataset.ground_truths, dataset.expected_contexts)):
                data.append({
                    "query": query,
                    "ground_truth": gt,
                    "expected_contexts": " | ".join(contexts),
                    "metadata": json.dumps(dataset.metadata)
                })
            df = pd.DataFrame(data)
            return df.to_csv(index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")


# Global instances
ragas_evaluator = RAGASEvaluator()
dataset_manager = EvaluationDatasetManager()
