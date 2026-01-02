"""
Article generation utilities for benchmark results.
Author: chandu
"""

from datetime import datetime
import os


def generate_tldr(results: dict) -> str:
    """Generate TL;DR section with key findings."""
    quality_results = results['quality']
    latency_results = results['latency']
    cost_results = results['cost']

    # Find best performers
    fastest_model = min(latency_results.items(), key=lambda x: x[1]['mean_ms'])
    best_quality_model = max(quality_results.items(), key=lambda x: x[1]['ndcg@10'])

    # Calculate value scores (quality per ms of latency)
    value_scores = {}
    for model_id in quality_results.keys():
        quality = quality_results[model_id]['ndcg@10']
        latency = latency_results[model_id]['mean_ms']
        value_scores[model_id] = quality / latency if latency > 0 else 0

    best_value_model = max(value_scores.items(), key=lambda x: x[1])

    # Find most cost-effective at 100K queries
    cost_effective = {}
    for model_id in cost_results.keys():
        quality = quality_results[model_id]['ndcg@10']
        cost_100k = cost_results[model_id]['cost_per_query'].get('100K', 1.0) * 100_000
        cost_effective[model_id] = quality / cost_100k if cost_100k > 0 else 0

    best_cost_effective = max(cost_effective.items(), key=lambda x: x[1])

    tldr = "## TL;DR\n\n"
    tldr += f"- **Fastest Model:** {fastest_model[0]} with {fastest_model[1]['mean_ms']:.2f}ms mean latency\n"
    tldr += f"- **Best Quality:** {best_quality_model[0]} with NDCG@10 of {best_quality_model[1]['ndcg@10']:.3f}\n"
    tldr += f"- **Best Value (Quality/Speed):** {best_value_model[0]} with score of {best_value_model[1]:.4f}\n"
    tldr += f"- **Most Cost-Effective:** {best_cost_effective[0]} for quality per dollar at 100K queries/month\n"
    tldr += f"- **Key Insight:** All models are local (no per-token costs), making them economical for high-volume deployments\n"

    return tldr


def generate_quality_table(results: dict) -> str:
    """Generate retrieval quality results table."""
    table = "\n## Retrieval Quality Results\n\n"
    table += "| Model | Recall@1 | Recall@5 | Recall@10 | NDCG@10 |\n"
    table += "|-------|----------|----------|-----------|----------|\n"

    for model_id, metrics in sorted(results['quality'].items()):
        table += f"| {model_id} | "
        table += f"{metrics['recall@1']:.3f} | "
        table += f"{metrics['recall@5']:.3f} | "
        table += f"{metrics['recall@10']:.3f} | "
        table += f"{metrics['ndcg@10']:.3f} |\n"

    return table


def generate_latency_table(results: dict) -> str:
    """Generate latency results table."""
    table = "\n## Latency Results\n\n"
    table += "| Model | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) | Std (ms) |\n"
    table += "|-------|-----------|-------------|----------|----------|----------|\n"

    for model_id, metrics in sorted(results['latency'].items()):
        table += f"| {model_id} | "
        table += f"{metrics['mean_ms']:.2f} | "
        table += f"{metrics['median_ms']:.2f} | "
        table += f"{metrics['p95_ms']:.2f} | "
        table += f"{metrics['p99_ms']:.2f} | "
        table += f"{metrics['std_ms']:.2f} |\n"

    return table


def generate_cost_table(results: dict) -> str:
    """Generate cost analysis table."""
    table = "\n## Cost Analysis\n\n"
    table += "| Model | Type | Monthly Infra | Cost/1K queries | Cost/100K queries | Cost/1M queries |\n"
    table += "|-------|------|---------------|-----------------|-------------------|-----------------|\n"

    for model_id, data in sorted(results['cost'].items()):
        table += f"| {model_id} | "
        table += f"{data['type']} | "
        table += f"${data['infrastructure_cost_monthly']:.2f} | "
        table += f"${data['cost_per_query'].get('1K', 0):.4f} | "
        table += f"${data['cost_per_query'].get('100K', 0) * 100_000:.2f} | "
        table += f"${data['cost_per_query'].get('1M', 0) * 1_000_000:.2f} |\n"

    table += "\n**Note:** All models are local, so costs are based on infrastructure only (AWS g4dn.xlarge GPU instance). "
    table += "Costs scale with query volume - if capacity is exceeded, multiple instances are needed.\n"

    return table


def generate_decision_matrix(results: dict) -> str:
    """Generate decision matrix with 'Choose if' guidance."""
    quality_results = results['quality']
    latency_results = results['latency']
    cost_results = results['cost']

    matrix = "\n## Decision Matrix\n\n"
    matrix += "*When to choose each model:*\n\n"

    for model_id in sorted(quality_results.keys()):
        quality = quality_results[model_id]
        latency = latency_results[model_id]
        cost = cost_results[model_id]

        matrix += f"### {model_id}\n\n"
        matrix += "**Choose if:**\n"

        # Generate specific criteria based on actual metrics
        if latency['mean_ms'] == min(latency_results[m]['mean_ms'] for m in latency_results):
            matrix += f"- You need the fastest inference speed ({latency['mean_ms']:.2f}ms mean latency)\n"

        if quality['ndcg@10'] == max(quality_results[m]['ndcg@10'] for m in quality_results):
            matrix += f"- You prioritize highest retrieval quality (NDCG@10: {quality['ndcg@10']:.3f})\n"

        if latency['mean_ms'] < 20:
            matrix += f"- You need sub-20ms response times for real-time applications\n"
        elif latency['mean_ms'] < 50:
            matrix += f"- You can accept moderate latency ({latency['mean_ms']:.1f}ms) for better quality\n"

        matrix += f"- You prefer local deployment for data privacy\n"
        matrix += f"- Your query volume is {int(cost['capacity_queries_per_month']):,}+ queries/month\n"

        matrix += "\n**Avoid if:**\n"

        if quality['ndcg@10'] < max(quality_results[m]['ndcg@10'] for m in quality_results):
            matrix += f"- Retrieval quality is critical (NDCG@10 {quality['ndcg@10']:.3f} vs best {max(quality_results[m]['ndcg@10'] for m in quality_results):.3f})\n"

        if latency['mean_ms'] > min(latency_results[m]['mean_ms'] for m in latency_results):
            matrix += f"- Minimizing latency is top priority (faster options available)\n"

        matrix += f"- You lack GPU infrastructure (requires GPU for optimal performance)\n"

        matrix += "\n---\n\n"

    return matrix


def generate_visualizations_section() -> str:
    """Generate visualizations section with image embeds."""
    section = "\n## Visualizations\n\n"

    charts = [
        ("quality_vs_latency.png", "Quality vs Speed Tradeoff"),
        ("recall_comparison.png", "Recall Metrics Comparison"),
        ("latency_boxplot.png", "Latency Distribution"),
        ("cost_comparison.png", "Cost at Different Scales"),
        ("model_comparison_heatmap.png", "Normalized Performance Comparison")
    ]

    for filename, caption in charts:
        section += f"### {caption}\n\n"
        section += f"![{caption}](visualizations/{filename})\n\n"

    return section


def generate_reproduction_instructions() -> str:
    """Generate reproduction instructions."""
    instructions = "\n## Reproduction Instructions\n\n"

    instructions += "### Prerequisites\n\n"
    instructions += "- Python 3.9 or higher\n"
    instructions += "- CUDA-capable GPU (optional but recommended for faster processing)\n"
    instructions += "- 8GB+ RAM\n"
    instructions += "- ~2GB disk space for models\n\n"

    instructions += "### Setup\n\n"
    instructions += "1. **Clone or download this repository**\n\n"
    instructions += "2. **Install dependencies:**\n"
    instructions += "```bash\n"
    instructions += "pip install -r requirements.txt\n"
    instructions += "```\n\n"

    instructions += "3. **Run the benchmark:**\n"
    instructions += "```bash\n"
    instructions += "python run_benchmarks.py\n"
    instructions += "```\n\n"

    instructions += "### Expected Runtime\n\n"
    instructions += "- **First run:** 30-45 minutes (includes model downloads)\n"
    instructions += "  - Model downloads: ~10-15 min (500MB total)\n"
    instructions += "  - Dataset generation: ~1-2 min\n"
    instructions += "  - Benchmarks: ~15-25 min\n"
    instructions += "  - Visualization: ~1 min\n\n"

    instructions += "- **Subsequent runs:** 15-20 minutes (models cached)\n\n"

    instructions += "### Output Files\n\n"
    instructions += "- `results/article.md` - This benchmark report\n"
    instructions += "- `results/raw_results.json` - Complete data in JSON format\n"
    instructions += "- `results/visualizations/` - All generated charts (PNG)\n\n"

    instructions += "### Customization\n\n"
    instructions += "Edit `benchmark_config.yaml` to:\n"
    instructions += "- Change dataset size (num_documents, num_queries)\n"
    instructions += "- Adjust latency measurement runs\n"
    instructions += "- Modify cost assumptions\n"
    instructions += "- Add/remove models\n\n"

    return instructions


def generate_metadata_section(results: dict) -> str:
    """Generate metadata section."""
    metadata = results.get('metadata', {})

    section = "\n## Benchmark Metadata\n\n"
    section += f"- **Timestamp:** {metadata.get('timestamp', 'N/A')}\n"
    section += f"- **Models Tested:** {len(results['quality'])}\n"
    section += f"- **Dataset Size:** {metadata.get('dataset_size', 'N/A')} documents\n"
    section += f"- **Number of Queries:** {metadata.get('num_queries', 'N/A')}\n"
    section += f"- **Latency Measurement Runs:** {metadata.get('latency_runs', 100)}\n\n"

    return section


def generate_raw_data_section(output_path: str) -> str:
    """Generate raw data section."""
    section = "\n## Raw Data\n\n"
    section += "Complete benchmark data is available in JSON format:\n\n"
    section += "- **File:** `raw_results.json`\n"
    section += "- **Location:** Same directory as this report\n"
    section += "- **Format:** Structured JSON with all metrics and metadata\n\n"
    section += "You can use this data for custom analysis or visualization.\n\n"

    return section


def generate_article(results: dict, output_path: str = "results/article.md"):
    """
    Generate complete markdown article.

    Args:
        results: Complete benchmark results dictionary
        output_path: Path to save article
    """
    print("\nGenerating article...")

    article_parts = []

    # Header
    article_parts.append("# Text Embeddings Benchmark Results\n")
    article_parts.append(f"**Author:** chandu  \n")
    article_parts.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}  \n")
    article_parts.append(f"**Models Tested:** {len(results['quality'])}  \n")
    article_parts.append("\n---\n")

    # TL;DR
    article_parts.append(generate_tldr(results))

    # Results Tables
    article_parts.append(generate_quality_table(results))
    article_parts.append(generate_latency_table(results))
    article_parts.append(generate_cost_table(results))

    # Decision Matrix
    article_parts.append(generate_decision_matrix(results))

    # Visualizations
    article_parts.append(generate_visualizations_section())

    # Reproduction Instructions
    article_parts.append(generate_reproduction_instructions())

    # Metadata
    article_parts.append(generate_metadata_section(results))

    # Raw Data
    article_parts.append(generate_raw_data_section(output_path))

    # Combine all parts
    article_content = '\n'.join(article_parts)

    # Write to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(article_content)

    print(f"[OK] Article saved to {output_path}")


if __name__ == "__main__":
    # Test the article generator
    print("Article generator module loaded successfully")
