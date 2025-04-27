import json
import os
from config import BenchmarkConfig

def save_results(results, filename="benchmark_results.json"):
    """Save benchmark results to JSON file"""
    results_path = os.path.join(BenchmarkConfig.LOG_DIR, filename)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")