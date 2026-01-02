"""
Main orchestrator for text embeddings benchmark.
Author: chandu

Run this script to execute the complete benchmark suite:
    python run_benchmarks.py
"""

import os
import json
from datetime import datetime
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.model_loader import load_models_from_config, load_config
from data.dataset_generator import load_or_generate_dataset
from benchmarks.retrieval_quality import run_retrieval_quality_benchmark
from benchmarks.latency import run_latency_benchmark
from benchmarks.cost_analysis import run_cost_analysis
from utils.metrics import validate_all_results
from utils.visualization import generate_all_visualizations
from utils.article_generator import generate_article


def print_banner():
    """Print welcome banner."""
    print("=" * 70)
    print(" " * 15 + "TEXT EMBEDDINGS BENCHMARK SUITE")
    print(" " * 25 + "Author: chandu")
    print("=" * 70)


def print_section(title: str):
    """Print section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main():
    """Main execution pipeline."""
    try:
        print_banner()

        # Step 1: Load configuration
        print_section("STEP 1/7: Loading Configuration")
        config_path = "benchmark_config.yaml"

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Please ensure benchmark_config.yaml exists in the current directory."
            )

        config = load_config(config_path)
        print(f"[OK] Configuration loaded from {config_path}")

        # Step 2: Load or generate dataset
        print_section("STEP 2/7: Loading Dataset")
        dataset = load_or_generate_dataset(config)
        print(f"[OK] Dataset ready: {len(dataset['documents'])} documents, {len(dataset['queries'])} queries")

        # Step 3: Load models
        print_section("STEP 3/7: Loading Models")
        models = load_models_from_config(config_path)
        print(f"\n[OK] Successfully loaded {len(models)} models: {', '.join(models.keys())}")

        # Get benchmark parameters
        benchmark_params = config.get('benchmark_parameters', {})
        latency_runs = benchmark_params.get('latency_runs', 100)
        batch_size = benchmark_params.get('batch_size', 32)

        # Step 4: Run retrieval quality benchmark
        print_section("STEP 4/7: Running Retrieval Quality Benchmark")
        print(f"Measuring Recall@1, Recall@5, Recall@10, and NDCG@10...")

        quality_results = {}
        for model_id, model in models.items():
            try:
                quality_results[model_id] = run_retrieval_quality_benchmark(
                    model, dataset, batch_size=batch_size
                )
            except Exception as e:
                print(f"\n[ERROR] Error running quality benchmark for {model_id}: {e}")
                print(f"  Skipping {model_id}...")
                continue

        print(f"\n[OK] Retrieval quality benchmark completed for {len(quality_results)} models")

        # Step 5: Run latency benchmark
        print_section("STEP 5/7: Running Latency Benchmark")
        print(f"Measuring Mean, P95, and P99 latency ({latency_runs} runs per model)...")

        latency_results = {}
        for model_id, model in models.items():
            if model_id not in quality_results:
                continue  # Skip models that failed quality benchmark

            try:
                latency_results[model_id] = run_latency_benchmark(
                    model, dataset, num_runs=latency_runs
                )
            except Exception as e:
                print(f"\n[ERROR] Error running latency benchmark for {model_id}: {e}")
                print(f"  Skipping {model_id}...")
                continue

        print(f"\n[OK] Latency benchmark completed for {len(latency_results)} models")

        # Step 6: Run cost analysis
        print_section("STEP 6/7: Running Cost Analysis")
        print(f"Calculating Total Cost of Ownership (TCO)...")

        cost_results = {}
        for model_id, model in models.items():
            if model_id not in latency_results:
                continue  # Skip models that failed previous benchmarks

            try:
                cost_results[model_id] = run_cost_analysis(
                    model, latency_results[model_id], config
                )
            except Exception as e:
                print(f"\n[ERROR] Error running cost analysis for {model_id}: {e}")
                print(f"  Skipping {model_id}...")
                continue

        print(f"\n[OK] Cost analysis completed for {len(cost_results)} models")

        # Validate results
        print("\nValidating results...")
        validate_all_results(quality_results, latency_results)

        # Step 7: Aggregate results
        print_section("STEP 7/7: Generating Outputs")

        all_results = {
            'quality': quality_results,
            'latency': latency_results,
            'cost': cost_results,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'config_file': config_path,
                'dataset_size': len(dataset['documents']),
                'num_queries': len(dataset['queries']),
                'models_tested': list(quality_results.keys()),
                'latency_runs': latency_runs,
                'batch_size': batch_size
            }
        }

        # Save raw results
        os.makedirs('results', exist_ok=True)
        raw_results_path = 'results/raw_results.json'

        with open(raw_results_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, default=str)

        print(f"[OK] Raw results saved to {raw_results_path}")

        # Generate visualizations
        generate_all_visualizations(all_results)

        # Generate article
        generate_article(all_results, 'results/article.md')

        # Print summary
        print("\n" + "=" * 70)
        print(" " * 20 + "BENCHMARK COMPLETE!")
        print("=" * 70)
        print(f"\nResults Summary:")
        print(f"   - Models tested: {len(quality_results)}")
        print(f"   - Dataset: {len(dataset['documents'])} documents, {len(dataset['queries'])} queries")
        print(f"   - Latency runs: {latency_runs} per model")
        print(f"\nOutput Files:")
        print(f"   - Article: results/article.md")
        print(f"   - Raw Data: results/raw_results.json")
        print(f"   - Visualizations: results/visualizations/")
        print(f"\nNext Steps:")
        print(f"   - Read the findings: results/article.md")
        print(f"   - View visualizations: results/visualizations/")
        print(f"   - Analyze raw data: results/raw_results.json")
        print("\n" + "=" * 70 + "\n")

        return 0

    except KeyboardInterrupt:
        print("\n\n[WARNING] Benchmark interrupted by user.")
        return 1

    except Exception as e:
        print(f"\n\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
