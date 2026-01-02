"""
Visualization utilities for benchmark results.
Author: chandu
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def plot_quality_vs_latency(results: dict, output_dir: str = "results/visualizations"):
    """
    Create scatter plot of quality vs latency.

    X-axis: Mean latency (ms)
    Y-axis: NDCG@10
    Points: Each model (labeled)
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    models = list(results['quality'].keys())
    x_values = [results['latency'][m]['mean_ms'] for m in models]
    y_values = [results['quality'][m]['ndcg@10'] for m in models]

    # Plot points
    ax.scatter(x_values, y_values, s=200, alpha=0.6, c='steelblue', edgecolors='black', linewidth=1.5)

    # Add model labels
    for i, model in enumerate(models):
        ax.annotate(model, (x_values[i], y_values[i]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

    ax.set_xlabel('Mean Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_ylabel('NDCG@10', fontsize=12, fontweight='bold')
    ax.set_title('Quality vs Speed Tradeoff', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)

    # Set y-axis limits to show full range
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'quality_vs_latency.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_recall_comparison(results: dict, output_dir: str = "results/visualizations"):
    """
    Create grouped bar chart for Recall@k comparison.

    X-axis: Models
    Y-axis: Recall (0-1)
    Bars: Recall@1, @5, @10 (different colors)
    """
    os.makedirs(output_dir, exist_ok=True)

    models = list(results['quality'].keys())
    recall_1 = [results['quality'][m]['recall@1'] for m in models]
    recall_5 = [results['quality'][m]['recall@5'] for m in models]
    recall_10 = [results['quality'][m]['recall@10'] for m in models]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width, recall_1, width, label='Recall@1', color='#FF6B6B', edgecolor='black')
    bars2 = ax.bar(x, recall_5, width, label='Recall@5', color='#4ECDC4', edgecolor='black')
    bars3 = ax.bar(x + width, recall_10, width, label='Recall@10', color='#45B7D1', edgecolor='black')

    ax.set_ylabel('Recall', fontsize=12, fontweight='bold')
    ax.set_title('Recall Comparison Across Models', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'recall_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_latency_boxplot(results: dict, output_dir: str = "results/visualizations"):
    """
    Create box plot showing latency distribution.

    X-axis: Models
    Y-axis: Latency (ms)
    Shows mean, P95, P99 as box/whiskers
    """
    os.makedirs(output_dir, exist_ok=True)

    models = list(results['latency'].keys())
    data_dict = {
        'Model': [],
        'Metric': [],
        'Latency (ms)': []
    }

    for model in models:
        metrics = results['latency'][model]
        data_dict['Model'].extend([model] * 3)
        data_dict['Metric'].extend(['Mean', 'P95', 'P99'])
        data_dict['Latency (ms)'].extend([
            metrics['mean_ms'],
            metrics['p95_ms'],
            metrics['p99_ms']
        ])

    df = pd.DataFrame(data_dict)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create grouped bar plot instead of box plot (since we have summary stats, not raw data)
    x = np.arange(len(models))
    width = 0.25

    mean_vals = [results['latency'][m]['mean_ms'] for m in models]
    p95_vals = [results['latency'][m]['p95_ms'] for m in models]
    p99_vals = [results['latency'][m]['p99_ms'] for m in models]

    bars1 = ax.bar(x - width, mean_vals, width, label='Mean', color='#95E1D3', edgecolor='black')
    bars2 = ax.bar(x, p95_vals, width, label='P95', color='#F38181', edgecolor='black')
    bars3 = ax.bar(x + width, p99_vals, width, label='P99', color='#AA96DA', edgecolor='black')

    ax.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Latency Distribution Across Models', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'latency_boxplot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_cost_comparison(results: dict, output_dir: str = "results/visualizations"):
    """
    Create line chart showing cost at different query volumes.

    X-axis: Query volume (1K, 10K, 100K, 1M/month)
    Y-axis: Monthly cost ($)
    Lines: Different models
    """
    os.makedirs(output_dir, exist_ok=True)

    models = list(results['cost'].keys())
    scales = ['1K', '10K', '100K', '1M']

    fig, ax = plt.subplots(figsize=(12, 6))

    for model in models:
        cost_data = results['cost'][model]['cost_per_query']
        # Convert cost per query to monthly cost
        monthly_costs = []
        for scale in scales:
            cost_per_query = cost_data.get(scale, 0)
            scale_num = int(scale.replace('K', '000').replace('M', '000000'))
            monthly_cost = cost_per_query * scale_num
            monthly_costs.append(monthly_cost)

        ax.plot(scales, monthly_costs, marker='o', linewidth=2, markersize=8, label=model)

    ax.set_xlabel('Query Volume (queries/month)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Monthly Cost ($)', fontsize=12, fontweight='bold')
    ax.set_title('Cost Comparison at Different Scales', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'cost_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_model_comparison_heatmap(results: dict, output_dir: str = "results/visualizations"):
    """
    Create heatmap showing normalized model performance.

    Rows: Models
    Columns: Normalized metrics (quality, speed, efficiency)
    Color: Green (good) to red (poor)
    """
    os.makedirs(output_dir, exist_ok=True)

    models = list(results['quality'].keys())

    # Collect metrics
    ndcg_scores = [results['quality'][m]['ndcg@10'] for m in models]
    latencies = [results['latency'][m]['mean_ms'] for m in models]

    # Normalize metrics (higher is better)
    # For latency, invert so lower latency = higher score
    max_latency = max(latencies)
    speed_scores = [1 - (lat / max_latency) for lat in latencies]

    # Create efficiency score (quality per ms)
    efficiency_scores = [ndcg / lat for ndcg, lat in zip(ndcg_scores, latencies)]
    max_efficiency = max(efficiency_scores)
    efficiency_scores = [eff / max_efficiency for eff in efficiency_scores]

    # Create dataframe
    data = {
        'NDCG@10\n(Quality)': ndcg_scores,
        'Speed\n(Inverse Latency)': speed_scores,
        'Efficiency\n(Quality/ms)': efficiency_scores
    }

    df = pd.DataFrame(data, index=models)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create heatmap
    sns.heatmap(df, annot=True, fmt='.3f', cmap='RdYlGn', center=0.5,
                cbar_kws={'label': 'Normalized Score'},
                linewidths=1, linecolor='black', ax=ax)

    ax.set_title('Model Performance Comparison (Normalized)', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_xlabel('')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'model_comparison_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def generate_all_visualizations(results: dict, output_dir: str = "results/visualizations"):
    """
    Generate all visualization charts.

    Args:
        results: Complete benchmark results dictionary
        output_dir: Directory to save visualizations
    """
    print("\nGenerating visualizations...")
    os.makedirs(output_dir, exist_ok=True)

    plot_quality_vs_latency(results, output_dir)
    plot_recall_comparison(results, output_dir)
    plot_latency_boxplot(results, output_dir)
    plot_cost_comparison(results, output_dir)
    plot_model_comparison_heatmap(results, output_dir)

    print(f"\n[OK] All visualizations saved to {output_dir}/")


if __name__ == "__main__":
    # Test the visualization module
    print("Visualization module loaded successfully")
