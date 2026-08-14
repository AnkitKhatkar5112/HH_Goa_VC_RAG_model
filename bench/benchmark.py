"""Latency benchmark script.

Runs 100+ diverse queries against the deployed API and reports
P50/P70/P100 per stage and end-to-end.

Usage:
    python -m bench.benchmark --target http://localhost:8000 --queries 100
    python -m bench.benchmark --target https://your-app.onrender.com --queries 150
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import random

import httpx
import numpy as np
import pandas as pd
from tabulate import tabulate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_eval_queries(data_dir: str = "data/processed", max_queries: int = 150) -> list[dict]:
    """Load evaluation queries from the processed dataset.

    Falls back to built-in sample queries if eval data is not available.
    """
    eval_path = os.path.join(data_dir, "eval_pairs.jsonl")
    queries = []

    if os.path.exists(eval_path):
        with open(eval_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                queries.append({
                    "query": data["query"],
                    "language": data["language"],
                })
                if len(queries) >= max_queries * 3:
                    break

        # Stratified sample across languages
        by_lang = {}
        for q in queries:
            by_lang.setdefault(q["language"], []).append(q)

        sampled = []
        per_lang = max_queries // len(by_lang) if by_lang else max_queries
        random.seed(42)
        for lang, lang_queries in by_lang.items():
            random.shuffle(lang_queries)
            sampled.extend(lang_queries[:per_lang])

        random.shuffle(sampled)
        return sampled[:max_queries]

    # Fallback: sample queries
    return [
        {"query": "What is the capital of India?", "language": "en"},
        {"query": "How does photosynthesis work?", "language": "en"},
        {"query": "भारत की राजधानी क्या है?", "language": "hi"},
        {"query": "प्रकाश संश्लेषण कैसे काम करता है?", "language": "hi"},
        {"query": "இந்தியாவின் தலைநகரம் என்ன?", "language": "ta"},
    ] * 20  # Repeat to get ~100


def run_benchmark(
    base_url: str,
    queries: list[dict],
    output_dir: str = "bench/results",
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Run the latency benchmark.

    Args:
        base_url: Base URL of the API.
        queries: List of query dicts with 'query' and 'language'.
        output_dir: Directory for output files.
        timeout: Request timeout in seconds.

    Returns:
        DataFrame with per-query latency data.
    """
    os.makedirs(output_dir, exist_ok=True)
    url = f"{base_url.rstrip('/')}/query/text"

    results = []
    print(f"\n{'='*60}")
    print(f"Latency Benchmark: {len(queries)} queries → {base_url}")
    print(f"{'='*60}\n")

    with httpx.Client(timeout=timeout) as client:
        for i, q in enumerate(queries, 1):
            try:
                start = time.perf_counter()
                resp = client.post(url, json={"query": q["query"], "language": q["language"]})
                wall_time_ms = (time.perf_counter() - start) * 1000
                resp.raise_for_status()
                data = resp.json()

                latency = data.get("latency", {})
                results.append({
                    "query_idx": i,
                    "language": q["language"],
                    "stt_ms": latency.get("stt_ms", 0),
                    "retrieval_ms": latency.get("retrieval_ms", 0),
                    "generation_ms": latency.get("generation_ms", 0),
                    "guardrails_ms": latency.get("guardrails_ms", 0),
                    "total_ms": latency.get("total_ms", 0),
                    "wall_time_ms": round(wall_time_ms, 2),
                    "status": "ok",
                })

                status_char = "." if i % 10 != 0 else f" [{i}/{len(queries)}]\n"
                print(status_char, end="", flush=True)

            except Exception as e:
                results.append({
                    "query_idx": i,
                    "language": q["language"],
                    "stt_ms": 0,
                    "retrieval_ms": 0,
                    "generation_ms": 0,
                    "guardrails_ms": 0,
                    "total_ms": 0,
                    "wall_time_ms": 0,
                    "status": f"error: {str(e)[:50]}",
                })
                print("E", end="", flush=True)

            time.sleep(0.1)  # Rate limiting

    print(f"\n\nCompleted {len(results)} queries.")

    df = pd.DataFrame(results)

    # Save raw CSV
    csv_path = os.path.join(output_dir, "latency_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved raw results to {csv_path}")

    return df


def generate_report(df: pd.DataFrame, output_dir: str = "bench/results") -> str:
    """Generate latency summary report.

    Args:
        df: DataFrame with benchmark results.
        output_dir: Directory for output files.

    Returns:
        Formatted markdown summary.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Filter successful queries
    ok = df[df["status"] == "ok"]
    if len(ok) == 0:
        print("No successful queries to report on.")
        return ""

    stages = ["stt_ms", "retrieval_ms", "generation_ms", "guardrails_ms", "total_ms"]
    stage_labels = ["STT", "Retrieval", "Generation", "Guardrails", "End-to-End"]

    # Compute percentiles
    rows = []
    for stage, label in zip(stages, stage_labels):
        values = ok[stage].values
        # Skip stages with all zeros
        if np.max(values) == 0:
            continue
        rows.append([
            label,
            f"{np.percentile(values, 50):.1f}",
            f"{np.percentile(values, 70):.1f}",
            f"{np.percentile(values, 100):.1f}",
            f"{np.mean(values):.1f}",
        ])

    headers = ["Stage", "P50 (ms)", "P70 (ms)", "P100 (ms)", "Mean (ms)"]
    table = tabulate(rows, headers=headers, tablefmt="github")

    # Summary text
    retrieval_p50 = np.percentile(ok["retrieval_ms"].values, 50)
    total_p50 = np.percentile(ok["total_ms"].values, 50)

    summary = f"""# Latency Benchmark Report

**Queries**: {len(ok)} successful out of {len(df)} total
**Target**: Deployed API

## Stage-by-Stage Latency

{table}

## Key Findings

- **Retrieval P50: {retrieval_p50:.1f} ms** {'✓ Under 200ms target' if retrieval_p50 < 200 else '⚠ Above 200ms target'}
- **End-to-End P50: {total_p50:.1f} ms**
- End-to-end latency includes STT, retrieval, LLM generation, and guardrail checks.
- The retrieval component (chunk lookup + vector search) is the part under direct control
  and is the component the <200ms requirement targets.
"""

    # Save markdown
    md_path = os.path.join(output_dir, "latency_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"\nSaved summary to {md_path}")
    print(f"\n{table}\n")

    # Generate chart
    _generate_chart(ok, output_dir)

    return summary


def _generate_chart(df: pd.DataFrame, output_dir: str) -> None:
    """Generate latency visualization chart."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('#0a0a0f')

        stages = {
            "Retrieval": df["retrieval_ms"].values,
            "Generation": df["generation_ms"].values,
            "Guardrails": df["guardrails_ms"].values,
        }
        # Filter out zero-only stages
        stages = {k: v for k, v in stages.items() if np.max(v) > 0}

        colors = ['#34d399', '#c084fc', '#fbbf24', '#818cf8']

        # Box plot
        ax1 = axes[0]
        ax1.set_facecolor('#12121a')
        bp = ax1.boxplot(stages.values(), labels=stages.keys(),
                         patch_artist=True, widths=0.5)
        for patch, color in zip(bp['boxes'], colors[:len(stages)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        for element in ['whiskers', 'caps', 'medians']:
            for line in bp[element]:
                line.set_color('#e8e8f0')
        for flier in bp['fliers']:
            flier.set_markeredgecolor('#8888a0')
        ax1.set_ylabel('Latency (ms)', color='#e8e8f0')
        ax1.set_title('Latency Distribution by Stage', color='#e8e8f0', fontweight='bold')
        ax1.tick_params(colors='#8888a0')
        ax1.spines['bottom'].set_color('#333')
        ax1.spines['left'].set_color('#333')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # Percentile bar chart
        ax2 = axes[1]
        ax2.set_facecolor('#12121a')
        percentiles = ['P50', 'P70', 'P100']
        x = np.arange(len(percentiles))
        width = 0.2
        for i, (name, vals) in enumerate(stages.items()):
            p_vals = [np.percentile(vals, p) for p in [50, 70, 100]]
            ax2.bar(x + i * width, p_vals, width, label=name, color=colors[i], alpha=0.7)
        ax2.set_xticks(x + width * (len(stages) - 1) / 2)
        ax2.set_xticklabels(percentiles)
        ax2.set_ylabel('Latency (ms)', color='#e8e8f0')
        ax2.set_title('Latency Percentiles', color='#e8e8f0', fontweight='bold')
        ax2.legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='#e8e8f0')
        ax2.tick_params(colors='#8888a0')
        ax2.spines['bottom'].set_color('#333')
        ax2.spines['left'].set_color('#333')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        plt.tight_layout()
        chart_path = os.path.join(output_dir, "latency_chart.png")
        plt.savefig(chart_path, dpi=150, facecolor='#0a0a0f', bbox_inches='tight')
        plt.close()
        print(f"Saved chart to {chart_path}")

    except ImportError as e:
        print(f"Chart generation skipped (missing dependency): {e}")


def main():
    parser = argparse.ArgumentParser(description="Run latency benchmark")
    parser.add_argument("--target", type=str, default="http://localhost:8000",
                        help="Base URL of the API")
    parser.add_argument("--queries", type=int, default=100,
                        help="Number of queries to run")
    parser.add_argument("--output-dir", type=str, default="bench/results",
                        help="Directory for output files")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Directory with eval data")
    args = parser.parse_args()

    queries = load_eval_queries(args.data_dir, args.queries)
    print(f"Loaded {len(queries)} evaluation queries")

    df = run_benchmark(args.target, queries, args.output_dir)
    generate_report(df, args.output_dir)


if __name__ == "__main__":
    main()
