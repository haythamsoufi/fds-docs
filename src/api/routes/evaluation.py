"""Evaluation routes for RAG system quality assessment."""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from src.services.evaluation_service import ragas_evaluator, dataset_manager, EvaluationResult, EvaluationDataset

router = APIRouter()


class EvaluationRequest(BaseModel):
    query: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None


class BatchEvaluationRequest(BaseModel):
    queries: List[str]
    answers: List[str]
    contexts_list: List[List[str]]
    ground_truths: Optional[List[Optional[str]]] = None


class EvaluationResponse(BaseModel):
    result: EvaluationResult
    status: str = "success"


class BatchEvaluationResponse(BaseModel):
    results: List[EvaluationResult]
    overall_scores: Dict[str, float]
    status: str = "success"


class DatasetResponse(BaseModel):
    datasets: List[str]
    status: str = "success"


@router.get("/status")
async def get_evaluation_status() -> Dict[str, Any]:
    """Get evaluation service status."""
    return ragas_evaluator.get_evaluation_status()


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_single_query(request: EvaluationRequest) -> EvaluationResponse:
    """Evaluate a single query-answer pair."""
    try:
        result = await ragas_evaluator.evaluate_query(
            query=request.query,
            answer=request.answer,
            contexts=request.contexts,
            ground_truth=request.ground_truth
        )
        
        return EvaluationResponse(result=result, status="success")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate-batch", response_model=BatchEvaluationResponse)
async def evaluate_batch_queries(request: BatchEvaluationRequest) -> BatchEvaluationResponse:
    """Evaluate a batch of query-answer pairs."""
    try:
        if len(request.queries) != len(request.answers) or len(request.queries) != len(request.contexts_list):
            raise HTTPException(
                status_code=400, 
                detail="Queries, answers, and contexts lists must have the same length"
            )
        
        if request.ground_truths and len(request.ground_truths) != len(request.queries):
            raise HTTPException(
                status_code=400,
                detail="Ground truths list must have the same length as queries"
            )
        
        results = await ragas_evaluator.evaluate_batch(
            queries=request.queries,
            answers=request.answers,
            contexts_list=request.contexts_list,
            ground_truths=request.ground_truths
        )
        
        overall_scores = ragas_evaluator.calculate_overall_scores(results)
        
        return BatchEvaluationResponse(
            results=results,
            overall_scores=overall_scores,
            status="success"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets", response_model=DatasetResponse)
async def list_datasets() -> DatasetResponse:
    """List available evaluation datasets."""
    try:
        datasets = dataset_manager.list_datasets()
        return DatasetResponse(datasets=datasets, status="success")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets/{dataset_name}")
async def get_dataset(dataset_name: str) -> Dict[str, Any]:
    """Get evaluation dataset by name."""
    try:
        dataset = dataset_manager.get_dataset(dataset_name)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
        
        return {
            "name": dataset.name,
            "description": dataset.description,
            "queries": dataset.queries,
            "ground_truths": dataset.ground_truths,
            "expected_contexts": dataset.expected_contexts,
            "metadata": dataset.metadata,
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/datasets/{dataset_name}/evaluate")
async def evaluate_dataset(
    dataset_name: str,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Evaluate a dataset against the current RAG system."""
    try:
        dataset = dataset_manager.get_dataset(dataset_name)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
        
        # Schedule evaluation in background
        background_tasks.add_task(_evaluate_dataset_task, dataset)
        
        return {
            "message": f"Evaluation of dataset '{dataset_name}' started",
            "dataset_size": len(dataset.queries),
            "status": "processing"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets/{dataset_name}/export")
async def export_dataset(
    dataset_name: str,
    format: str = "json"
) -> Dict[str, Any]:
    """Export evaluation dataset."""
    try:
        if format not in ["json", "csv"]:
            raise HTTPException(status_code=400, detail="Format must be 'json' or 'csv'")
        
        exported_data = dataset_manager.export_dataset(dataset_name, format)
        if not exported_data:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
        
        return {
            "data": exported_data,
            "format": format,
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results/{evaluation_id}")
async def get_evaluation_results(evaluation_id: str) -> Dict[str, Any]:
    """Get evaluation results by ID."""
    try:
        # This would typically fetch from a database or file system
        # For now, return a placeholder response
        return {
            "evaluation_id": evaluation_id,
            "status": "not_implemented",
            "message": "Result storage not implemented yet"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/results/export")
async def export_evaluation_results(
    results: List[EvaluationResult],
    format: str = "json"
) -> Dict[str, Any]:
    """Export evaluation results."""
    try:
        if format not in ["json", "csv"]:
            raise HTTPException(status_code=400, detail="Format must be 'json' or 'csv'")
        
        exported_data = ragas_evaluator.export_results(results, format)
        
        return {
            "data": exported_data,
            "format": format,
            "count": len(results),
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _evaluate_dataset_task(dataset: EvaluationDataset):
    """Background task to evaluate a dataset."""
    try:
        from src.services.query_service import query_service
        
        # Evaluate each query in the dataset
        results = []
        for i, (query, ground_truth, expected_contexts) in enumerate(zip(
            dataset.queries, 
            dataset.ground_truths, 
            dataset.expected_contexts
        )):
            try:
                # Query the RAG system
                response = await query_service.searchDocuments({
                    "query": query,
                    "max_results": 5
                })
                
                # Extract contexts
                contexts = [chunk.chunk.content for chunk in response.retrieved_chunks]
                
                # Evaluate the response
                result = await ragas_evaluator.evaluate_query(
                    query=query,
                    answer=response.answer,
                    contexts=contexts,
                    ground_truth=ground_truth
                )
                
                results.append(result)
                
                # Log progress
                print(f"Evaluated query {i+1}/{len(dataset.queries)}: {query[:50]}...")
                
            except Exception as e:
                print(f"Error evaluating query {i+1}: {e}")
                continue
        
        # Calculate overall scores
        overall_scores = ragas_evaluator.calculate_overall_scores(results)
        
        # Export results
        exported_data = ragas_evaluator.export_results(results, "json")
        
        # Save results (in a real implementation, this would save to database)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results_{dataset.name}_{timestamp}.json"
        
        with open(f"data/evaluations/{filename}", "w") as f:
            f.write(exported_data)
        
        print(f"Dataset evaluation completed: {len(results)} queries evaluated")
        print(f"Results saved to: {filename}")
        print(f"Overall scores: {overall_scores}")
        
    except Exception as e:
        print(f"Dataset evaluation failed: {e}")
