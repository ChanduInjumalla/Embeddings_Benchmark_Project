"""
Metric calculation and validation utilities.
Author: chandu
"""

from typing import List, Set, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between two sets of vectors.

    Args:
        a: Array of shape (n, d) - query embeddings
        b: Array of shape (m, d) - document embeddings

    Returns:
        Similarity matrix of shape (n, m)
    """
    return sklearn_cosine_similarity(a, b)


def calculate_recall(retrieved: List[str], relevant: Set[str]) -> float:
    """
    Calculate recall: |retrieved ∩ relevant| / |relevant|

    Args:
        retrieved: List of retrieved document IDs
        relevant: Set of relevant document IDs

    Returns:
        Recall score between 0 and 1
    """
    if not relevant:
        return 0.0

    retrieved_set = set(retrieved)
    intersection = retrieved_set & relevant

    return len(intersection) / len(relevant)


def calculate_dcg(relevance_scores: List[float], k: int = 10) -> float:
    """
    Calculate Discounted Cumulative Gain.

    DCG@k = sum(rel_i / log2(i+1)) for i in 1..k

    Args:
        relevance_scores: List of relevance scores in ranked order
        k: Cutoff position

    Returns:
        DCG score
    """
    dcg = 0.0
    for i, rel in enumerate(relevance_scores[:k]):
        dcg += rel / np.log2(i + 2)  # i+2 because i is 0-indexed

    return dcg


def calculate_ndcg(ranked_docs: List[str], relevance_dict: Dict[str, int], k: int = 10) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain.

    NDCG@k = DCG@k / IDCG@k

    Args:
        ranked_docs: List of document IDs in ranked order
        relevance_dict: Dictionary mapping document IDs to relevance scores
        k: Cutoff position

    Returns:
        NDCG score between 0 and 1
    """
    # Get relevance scores for ranked documents
    relevance_scores = [relevance_dict.get(doc_id, 0) for doc_id in ranked_docs[:k]]

    # Calculate DCG
    dcg = calculate_dcg(relevance_scores, k)

    # Calculate IDCG (ideal DCG with perfect ranking)
    ideal_scores = sorted(relevance_dict.values(), reverse=True)[:k]
    idcg = calculate_dcg(ideal_scores, k)

    # Return NDCG
    if idcg == 0:
        return 0.0

    return dcg / idcg


def validate_recall(value: float, name: str = "Recall") -> None:
    """
    Validate that recall value is in [0, 1].

    Args:
        value: Recall value to validate
        name: Name of the metric (for error messages)

    Raises:
        ValueError: If value is outside valid range
    """
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} value {value} is outside valid range [0, 1]")


def validate_latency(value: float, name: str = "Latency") -> None:
    """
    Validate that latency value is positive.

    Args:
        value: Latency value to validate
        name: Name of the metric (for error messages)

    Raises:
        ValueError: If value is not positive
    """
    if value <= 0:
        raise ValueError(f"{name} value {value} must be positive")


def validate_quality_results(results: dict, model_name: str) -> None:
    """
    Validate retrieval quality results.

    Args:
        results: Dictionary with quality metrics
        model_name: Name of the model (for error messages)

    Raises:
        ValueError: If results are invalid
    """
    required_keys = ['recall@1', 'recall@5', 'recall@10', 'ndcg@10']

    for key in required_keys:
        if key not in results:
            raise ValueError(f"Missing required metric: {key}")

        value = results[key]
        validate_recall(value, f"{model_name} {key}")

    # Check monotonicity: recall@10 >= recall@5 >= recall@1
    if not (results['recall@1'] <= results['recall@5'] <= results['recall@10']):
        raise ValueError(
            f"{model_name}: Recall values not monotonic: "
            f"R@1={results['recall@1']:.3f}, "
            f"R@5={results['recall@5']:.3f}, "
            f"R@10={results['recall@10']:.3f}"
        )


def validate_latency_results(results: dict, model_name: str) -> None:
    """
    Validate latency results.

    Args:
        results: Dictionary with latency metrics
        model_name: Name of the model (for error messages)

    Raises:
        ValueError: If results are invalid
    """
    required_keys = ['mean_ms', 'p95_ms', 'p99_ms']

    for key in required_keys:
        if key not in results:
            raise ValueError(f"Missing required metric: {key}")

        value = results[key]
        validate_latency(value, f"{model_name} {key}")

    # Check ordering: p99 >= p95 >= mean
    if not (results['mean_ms'] <= results['p95_ms'] <= results['p99_ms']):
        raise ValueError(
            f"{model_name}: Latency percentiles not ordered: "
            f"mean={results['mean_ms']:.2f}ms, "
            f"P95={results['p95_ms']:.2f}ms, "
            f"P99={results['p99_ms']:.2f}ms"
        )


def validate_all_results(quality_results: dict, latency_results: dict) -> None:
    """
    Validate all benchmark results.

    Args:
        quality_results: Dictionary of quality results per model
        latency_results: Dictionary of latency results per model

    Raises:
        ValueError: If any results are invalid
    """
    print("\nValidating benchmark results...")

    # Validate quality results
    for model_name, results in quality_results.items():
        validate_quality_results(results, model_name)

    # Validate latency results
    for model_name, results in latency_results.items():
        validate_latency_results(results, model_name)

    print("[OK] All results passed validation")
