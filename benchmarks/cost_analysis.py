"""
Cost analysis benchmark - calculates TCO for local models.
Author: chandu
"""

from models.model_wrapper import EmbeddingModel


def run_cost_analysis(
    model: EmbeddingModel,
    latency_results: dict,
    config: dict
) -> dict:
    """
    Run cost analysis for a model.

    For local models, calculates Total Cost of Ownership (TCO) based on
    infrastructure costs and throughput capacity.

    Args:
        model: EmbeddingModel instance
        latency_results: Results from latency benchmark
        config: Configuration dictionary with cost assumptions

    Returns:
        Dictionary with cost breakdown
    """
    model_name = model.get_name()
    print(f"\n  Running cost analysis for {model_name}...")

    # Get benchmark parameters
    benchmark_params = config.get('benchmark_parameters', {})
    gpu_hourly_cost = benchmark_params.get('gpu_instance_hourly_cost', 0.526)  # AWS g4dn.xlarge
    hours_per_month = benchmark_params.get('hours_per_month', 730)

    # Calculate monthly infrastructure cost
    monthly_infra_cost = gpu_hourly_cost * hours_per_month

    # Calculate throughput capacity
    mean_latency_ms = latency_results.get('mean_ms', 10.0)
    queries_per_second = 1000.0 / mean_latency_ms
    queries_per_month = queries_per_second * 60 * 60 * 24 * 30

    # Calculate cost per query at different scales
    scales = [1_000, 10_000, 100_000, 1_000_000]
    cost_per_query = {}

    for scale in scales:
        scale_name = f"{scale//1000}K" if scale < 1_000_000 else f"{scale//1_000_000}M"

        # If scale exceeds capacity, would need multiple instances
        num_instances = max(1, scale / queries_per_month)
        actual_monthly_cost = monthly_infra_cost * num_instances
        cost_per_query[scale_name] = actual_monthly_cost / scale

    results = {
        'type': 'local',
        'model_name': model_name,
        'cost_per_million_tokens': 0.0,  # Local models have no per-token cost
        'infrastructure_cost_monthly': monthly_infra_cost,
        'gpu_hourly_cost': gpu_hourly_cost,
        'queries_per_second': queries_per_second,
        'capacity_queries_per_month': queries_per_month,
        'cost_per_query': cost_per_query,
        'assumptions': {
            'instance_type': 'AWS g4dn.xlarge',
            'hourly_cost': gpu_hourly_cost,
            'hours_per_month': hours_per_month,
            'based_on_latency_ms': mean_latency_ms
        }
    }

    print(f"    Monthly infrastructure: ${monthly_infra_cost:.2f}")
    print(f"    Capacity: {queries_per_month:,.0f} queries/month")
    print(f"    Cost per 1K queries: ${cost_per_query.get('1K', 0):.4f}")
    print(f"    Cost per 1M queries: ${cost_per_query.get('1M', 0):.2f}")

    return results


if __name__ == "__main__":
    # Test the cost analysis
    print("Cost analysis module loaded successfully")
