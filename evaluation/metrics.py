"""
Standalone metric computation functions for validation and benchmarking.
"""

from typing import Dict, List, Any


def precision_at_k(predictions: List[float], labels: List[int], k: int) -> float:
    """Calculate precision at k."""
    if not predictions or k <= 0:
        return 0.0
    # Pair scores with true labels
    paired = sorted(zip(predictions, labels), key=lambda x: -x[0])
    top_k = paired[:k]
    tp = sum(1 for _, l in top_k if l == 1)
    return tp / k


def recall_at_k(predictions: List[float], labels: List[int], k: int) -> float:
    """Calculate recall at k."""
    if not predictions or k <= 0:
        return 0.0
    total_positives = sum(labels)
    if total_positives == 0:
        return 1.0
    paired = sorted(zip(predictions, labels), key=lambda x: -x[0])
    top_k = paired[:k]
    tp = sum(1 for _, l in top_k if l == 1)
    return tp / total_positives


def average_precision(predictions: List[float], labels: List[int]) -> float:
    """Calculate average precision."""
    if not predictions or len(predictions) != len(labels):
        return 0.0
    paired = sorted(enumerate(zip(predictions, labels)), key=lambda x: -x[1][0])
    
    num_positives = 0
    sum_precisions = 0.0
    
    for idx, (_, (_, label)) in enumerate(paired):
        if label == 1:
            num_positives += 1
            sum_precisions += num_positives / (idx + 1)
            
    if num_positives == 0:
        return 0.0
        
    return sum_precisions / num_positives


def hallucination_rate(claims: List[str], evidence_log: List[Dict[str, Any]]) -> float:
    """Calculate hallucination rate directly from claim lists and evidence logs."""
    if not evidence_log:
        return 0.0
    hallucinated = sum(1 for item in evidence_log if item.get("hallucinated", False))
    return hallucinated / len(evidence_log)


def grounding_accuracy(evidence_log: List[Dict[str, Any]]) -> float:
    """Calculate grounding accuracy (percentage of claims verified by visual or metadata)."""
    if not evidence_log:
        return 1.0
    grounded = sum(1 for item in evidence_log if item.get("source_type") in ["visual", "history", "prior"])
    return grounded / len(evidence_log)
