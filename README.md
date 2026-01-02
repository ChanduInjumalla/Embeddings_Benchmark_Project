# Text Embeddings Benchmark

**Author:** chandu

Comprehensive benchmark comparing text embedding models across retrieval quality, latency, and cost dimensions. This benchmark evaluates 3 local sentence-transformer models on real retrieval tasks with detailed performance analysis.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run benchmark
python run_benchmarks.py
```

The benchmark will:
1. Generate a synthetic test dataset (500 documents, 100 queries)
2. Load 3 embedding models
3. Measure retrieval quality (Recall@k, NDCG)
4. Measure latency (Mean, P95, P99)
5. Calculate cost analysis (TCO)
6. Generate visualizations
7. Create comprehensive report in `results/article.md`

## Models Tested

| Model | Type | Dimensions | Use Case |
|-------|------|------------|----------|
| **sentence-transformers/all-MiniLM-L6-v2** | Local | 384 | Fastest, lightweight, CPU-friendly |
| **BAAI/bge-base-en-v1.5** | Local | 768 | Balanced quality and speed |
| **BAAI/bge-large-en-v1.5** | Local | 1024 | Best retrieval quality |

All models are open-source and run locally (no API costs).

## Benchmarks

### 1. Retrieval Quality
Measures how well models find relevant documents:
- **Recall@1**: Fraction of queries where relevant doc is #1 result
- **Recall@5**: Fraction of queries where relevant doc is in top 5
- **Recall@10**: Fraction of queries where relevant doc is in top 10
- **NDCG@10**: Normalized Discounted Cumulative Gain (accounts for ranking)

### 2. Latency
Measures inference speed:
- **Mean**: Average encoding time per query
- **P95**: 95th percentile (SLA-critical)
- **P99**: 99th percentile (worst-case performance)

### 3. Cost Analysis
Calculates Total Cost of Ownership (TCO):
- **Infrastructure cost**: AWS GPU instance pricing
- **Throughput capacity**: Queries/month per instance
- **Cost per query**: At different scales (1K, 10K, 100K, 1M queries/month)

## Project Structure

```
interview/
├── benchmarks/              # Benchmark modules
│   ├── retrieval_quality.py   # Recall@k, NDCG@10
│   ├── latency.py             # Latency measurement
│   └── cost_analysis.py       # TCO calculation
├── data/                    # Dataset generation
│   ├── dataset_generator.py   # Synthetic corpus creator
│   └── datasets/              # Cached datasets
├── models/                  # Model loading
│   ├── model_wrapper.py       # Unified interface
│   └── model_loader.py        # Config-based loading
├── utils/                   # Utilities
│   ├── metrics.py             # Metric calculations
│   ├── visualization.py       # Chart generation
│   └── article_generator.py   # Report creation
├── results/                 # Output directory
│   ├── article.md             # Main report
│   ├── raw_results.json       # Complete data
│   └── visualizations/        # Charts (PNG)
├── benchmark_config.yaml    # Configuration
├── run_benchmarks.py        # Main orchestrator
├── requirements.txt         # Dependencies
└── README.md                # This file
```

## Prerequisites

- **Python:** 3.9 or higher
- **RAM:** 8GB+ recommended
- **GPU:** CUDA-capable GPU optional (faster processing, but CPU works)
- **Disk:** ~2GB for model weights
- **OS:** Windows, Linux, or macOS

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `sentence-transformers` - Model loading and inference
- `torch` - Deep learning framework
- `numpy`, `scipy` - Numerical computation
- `pandas` - Data handling
- `matplotlib`, `seaborn` - Visualization
- `pyyaml` - Configuration parsing
- `tqdm` - Progress bars

### 2. Verify Installation

```bash
python -c "import sentence_transformers; print('✓ Installation successful')"
```

## Usage

### Run Full Benchmark

```bash
python run_benchmarks.py
```

**Expected runtime:**
- First run: 30-45 minutes (downloads models ~500MB)
- Subsequent runs: 15-20 minutes (models cached)

**Output:**
- `results/article.md` - Comprehensive report with findings
- `results/raw_results.json` - Complete benchmark data (JSON)
- `results/visualizations/` - 5 charts (PNG)

### Run Individual Benchmarks

```bash
# Quality benchmark only
python -c "from benchmarks.retrieval_quality import *; print('Quality benchmark ready')"

# Latency benchmark only
python -c "from benchmarks.latency import *; print('Latency benchmark ready')"

# Cost analysis only
python -c "from benchmarks.cost_analysis import *; print('Cost analysis ready')"
```

### Generate Test Data Only

```bash
python data/dataset_generator.py
```

## Configuration

Edit `benchmark_config.yaml` to customize:

```yaml
# Dataset parameters
dataset:
  num_documents: 500    # Number of documents in corpus
  num_queries: 100      # Number of test queries
  seed: 42              # Random seed for reproducibility

# Benchmark parameters
benchmark_parameters:
  latency_runs: 100           # Number of latency measurements
  latency_warmup_runs: 5      # Warmup runs (excluded from stats)
  batch_size: 32              # Encoding batch size
  gpu_instance_hourly_cost: 0.526  # AWS g4dn.xlarge cost
```

## Results

After running the benchmark, check:

### 1. Article Report (`results/article.md`)

Comprehensive report with:
- **TL;DR**: Key findings in 3-5 bullet points
- **Results tables**: All metrics for all models
- **Decision matrix**: "Choose X if..." guidance
- **Visualizations**: 5 charts
- **Reproduction instructions**: How to run

### 2. Raw Data (`results/raw_results.json`)

Complete data in JSON format for custom analysis:
```json
{
  "quality": {
    "minilm": {"recall@1": 0.45, "recall@5": 0.72, ...},
    ...
  },
  "latency": {...},
  "cost": {...},
  "metadata": {...}
}
```

### 3. Visualizations (`results/visualizations/`)

5 charts (PNG, 300 DPI):
1. **quality_vs_latency.png** - Quality/speed tradeoff scatter plot
2. **recall_comparison.png** - Recall@k bar chart
3. **latency_boxplot.png** - Latency distribution
4. **cost_comparison.png** - Cost at different scales
5. **model_comparison_heatmap.png** - Normalized performance heatmap

## Interpreting Results

### Retrieval Quality Metrics

- **Recall@k ∈ [0, 1]**: Higher is better
  - 0.5 = 50% of queries found relevant doc in top k
  - 1.0 = 100% perfect retrieval

- **NDCG@10 ∈ [0, 1]**: Higher is better
  - Accounts for ranking position
  - 1.0 = perfect ranking
  - 0.7+ = good retrieval quality

### Latency Metrics

- **Mean latency**: Typical performance
- **P95**: 95% of queries complete within this time
- **P99**: Worst-case for SLA compliance

Good targets:
- Real-time apps: <20ms mean
- Interactive apps: <50ms mean
- Batch processing: <200ms mean

### Cost Metrics

For local models, costs are:
- **Fixed**: Infrastructure cost (GPU instance)
- **Variable**: Number of instances needed for capacity

Break-even analysis:
- Low volume (<10K/month): Single instance amortizes well
- High volume (>100K/month): May need multiple instances

## Development

### Adding New Models

1. Edit `benchmark_config.yaml`:

```yaml
models:
  - id: "my-model"
    type: "local"
    model_name: "your/model-name"
    dimensions: 768
```

2. Run benchmark:

```bash
python run_benchmarks.py
```

### Extending Benchmarks

Create new benchmark in `benchmarks/`:

```python
def run_my_benchmark(model, dataset, config):
    # Your benchmark logic
    return results
```

Add to `run_benchmarks.py` orchestrator.

### Testing

Quick test with small dataset:

1. Edit `benchmark_config.yaml`:
```yaml
dataset:
  num_documents: 50   # Instead of 500
  num_queries: 10     # Instead of 100
```

2. Run:
```bash
python run_benchmarks.py
```

Should complete in <5 minutes.

## Troubleshooting

### Models Download Slowly

Models are cached in `~/.cache/huggingface/`. First run downloads ~500MB.

To pre-download:
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

### Out of Memory (OOM)

Reduce batch size in `benchmark_config.yaml`:
```yaml
benchmark_parameters:
  batch_size: 16  # Default is 32
```

### GPU Not Detected

Check CUDA installation:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

Models work on CPU (slower but functional).

### Permission Errors (Windows)

Run terminal as Administrator or use a directory without OneDrive sync.

## Benchmarking Best Practices

1. **Close other applications** - Reduce background interference
2. **Consistent environment** - Same hardware for fair comparison
3. **Multiple runs** - Run 2-3 times, report average (currently single run)
4. **Representative data** - Ensure test queries match real use case
5. **Document setup** - Record hardware specs in results

## Performance Tips

- **GPU**: Use CUDA GPU for 5-10x faster encoding
- **Batch size**: Larger batches = better GPU utilization (up to memory limit)
- **Caching**: Dataset and models are cached automatically
- **Parallel**: Can run multiple benchmarks in parallel (modify code)

## Citation

If you use this benchmark in your work:

```
Text Embeddings Benchmark
Author: chandu
Year: 2025
Models: sentence-transformers/all-MiniLM-L6-v2, BAAI/bge-base-en-v1.5, BAAI/bge-large-en-v1.5
```

## License

This benchmark code is provided as-is for educational and evaluation purposes.

Model licenses:
- sentence-transformers/all-MiniLM-L6-v2: Apache 2.0
- BAAI/bge-base-en-v1.5: MIT
- BAAI/bge-large-en-v1.5: MIT

## Acknowledgments

- **Sentence Transformers** - Model framework
- **BAAI** - BGE model series
- **HuggingFace** - Model hosting

## Conclusion

This project provides a structured and practical framework for benchmarking and evaluating different text embedding models based on their retrieval quality and similarity performance. By implementing standardized evaluation metrics and a reusable benchmarking pipeline, the project enables objective comparison of embedding models under consistent conditions.

Through this benchmark, we gain clear insights into how various embedding models perform in real-world retrieval scenarios, highlighting their strengths, limitations, and suitability for downstream tasks such as semantic search, document retrieval, and recommendation systems. The modular design of the codebase ensures scalability and makes it easy to integrate additional models, datasets, and evaluation metrics in the future.

Overall, this project serves as a reliable foundation for researchers, data scientists, and engineers to make informed decisions when selecting embedding models, while also promoting reproducibility and transparency in embedding evaluation workflows. It demonstrates the importance of empirical benchmarking in building robust and high-quality NLP systems.

## Contact

**Author:** chandu

For issues or questions about this benchmark, refer to the generated `results/article.md` for detailed findings and reproduction instructions.

---

**Happy Benchmarking!** 🚀
