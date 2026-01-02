"""
Retrieval quality benchmark - measures Recall@k and NDCG@10.
Author: chandu
"""

from typing import Dict
import numpy as np
from tqdm import tqdm
from models.model_wrapper import EmbeddingModel
from utils.metrics import cosine_similarity, calculate_recall, calculate_ndcg


def run_retrieval_quality_benchmark(
    model: EmbeddingModel,
    dataset: dict,
    batch_size: int = 32
) -> dict:
    """
    Run retrieval quality benchmark on a model.

    Args:
        model: EmbeddingModel instance
        dataset: Dataset dictionary with documents and queries
        batch_size: Batch size for encoding

    Returns:
        Dictionary with Recall@1, Recall@5, Recall@10, NDCG@10
    """
    documents = dataset['documents']
    queries = dataset['queries']

    model_name = model.get_name()
    print(f"\n  Running retrieval quality benchmark for {model_name}...")

    # Step 1: Encode all documents
    print(f"    Encoding {len(documents)} documents...")
    doc_texts = [doc['text'] for doc in documents]
    doc_embeddings = model.encode(doc_texts, batch_size=batch_size)

    # Create mapping from doc_id to index
    doc_id_to_idx = {doc['id']: idx for idx, doc in enumerate(documents)}

    # Step 2: Encode all queries
    print(f"    Encoding {len(queries)} queries...")
    query_texts = [query['text'] for query in queries]
    query_embeddings = model.encode(query_texts, batch_size=batch_size)

    # Step 3: Compute similarity matrix
    print(f"    Computing similarities...")
    similarities = cosine_similarity(query_embeddings, doc_embeddings)

    # Step 4: Calculate metrics for each query
    recall_at_1 = []
    recall_at_5 = []
    recall_at_10 = []
    ndcg_at_10 = []

    for query_idx, query in enumerate(tqdm(queries, desc=f"    Evaluating queries", leave=False)):
        # Get similarity scores for this query
        query_sims = similarities[query_idx]

        # Rank documents by similarity (descending)
        ranked_indices = np.argsort(-query_sims)  # Negative for descending
        ranked_doc_ids = [documents[idx]['id'] for idx in ranked_indices]

        # Get ground truth relevant documents
        relevance_dict = query['relevance']
        relevant_doc_ids = set(relevance_dict.keys())

        # Calculate Recall@k
        recall_at_1.append(calculate_recall(ranked_doc_ids[:1], relevant_doc_ids))
        recall_at_5.append(calculate_recall(ranked_doc_ids[:5], relevant_doc_ids))
        recall_at_10.append(calculate_recall(ranked_doc_ids[:10], relevant_doc_ids))

        # Calculate NDCG@10
        ndcg_at_10.append(calculate_ndcg(ranked_doc_ids, relevance_dict, k=10))

    # Step 5: Average metrics across all queries
    results = {
        'recall@1': np.mean(recall_at_1),
        'recall@5': np.mean(recall_at_5),
        'recall@10': np.mean(recall_at_10),
        'ndcg@10': np.mean(ndcg_at_10)
    }

    print(f"    Results: Recall@1={results['recall@1']:.3f}, "
          f"Recall@5={results['recall@5']:.3f}, "
          f"Recall@10={results['recall@10']:.3f}, "
          f"NDCG@10={results['ndcg@10']:.3f}")

    return results


if __name__ == "__main__":
    # Test the benchmark
    print("Retrieval quality benchmark module loaded successfully")
