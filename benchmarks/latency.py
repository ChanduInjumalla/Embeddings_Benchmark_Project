"""
Latency benchmark - measures mean, P95, and P99 latency.
Author: chandu
"""

import time
import random
import numpy as np
from tqdm import tqdm
from models.model_wrapper import EmbeddingModel


def run_latency_benchmark(
    model: EmbeddingModel,
    dataset: dict,
    num_runs: int = 100,
    warmup_runs: int = 5
) -> dict:
    """
    Run latency benchmark on a model.

    Args:
        model: EmbeddingModel instance
        dataset: Dataset dictionary with queries
        num_runs: Number of measurement runs
        warmup_runs: Number of warmup runs to exclude

    Returns:
        Dictionary with mean_ms, p95_ms, p99_ms latency metrics
    """
    queries = dataset['queries']
    model_name = model.get_name()

    print(f"\n  Running latency benchmark for {model_name}...")

    # Sample texts for latency measurement
    sample_texts = [query['text'] for query in queries]
    if len(sample_texts) > num_runs:
        sample_texts = random.sample(sample_texts, num_runs)

    # Warmup phase
    print(f"    Warming up ({warmup_runs} runs)...")
    for i in range(warmup_runs):
        text = random.choice(sample_texts)
        _ = model.encode([text])

    # Measurement phase
    print(f"    Measuring latency ({num_runs} runs)...")
    latencies = []

    for text in tqdm(sample_texts[:num_runs], desc=f"    Timing", leave=False):
        start = time.perf_counter()
        _ = model.encode([text])
        end = time.perf_counter()

        latency_ms = (end - start) * 1000  # Convert to milliseconds
        latencies.append(latency_ms)

    # Calculate statistics
    latencies = np.array(latencies)

    results = {
        'mean_ms': float(np.mean(latencies)),
        'median_ms': float(np.median(latencies)),
        'std_ms': float(np.std(latencies)),
        'p95_ms': float(np.percentile(latencies, 95)),
        'p99_ms': float(np.percentile(latencies, 99)),
        'min_ms': float(np.min(latencies)),
        'max_ms': float(np.max(latencies))
    }

    print(f"    Results: Mean={results['mean_ms']:.2f}ms, "
          f"P95={results['p95_ms']:.2f}ms, "
          f"P99={results['p99_ms']:.2f}ms")

    return results


if __name__ == "__main__":
    # Test the benchmark
    print("Latency benchmark module loaded successfully")
