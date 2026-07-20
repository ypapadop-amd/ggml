#!/usr/bin/env python3
"""Plot bench-mul-mat-hsa results as a bundled bar graph.

Bars are bundled per (M, N, K, dtype) shape; within each bundle, one bar per
backend (and per input file/configuration, if more than one file is given).

Usage:
    ./plot_benchmarks.py results.json
    ./plot_benchmarks.py --labels baseline,tuned baseline.json tuned.json
    ./plot_benchmarks.py --output plot.png results.json

Generate input files with:
    ./bench-mul-mat-hsa --benchmark_format=json --benchmark_out=results.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

NAME_RE = re.compile(
    r"^bench_mul_mat<BackendType::(\w+),\s*(\w+)>/(\d+)/(\d+)/(\d+)(?:/real_time)?$"
)

DTYPE_LABELS = {
    "float": "F32",
    "ggml_bf16_t": "BF16",
}


def parse_benchmark_file(path):
    """Parse a Google Benchmark JSON output file.

    Returns a list of (backend, dtype, m, n, k, gflops, time_us) tuples. Runs
    that errored out (e.g. unsupported op) are skipped with a warning.
    """
    # ns -> us divisors by time_unit, so time is always reported in microseconds.
    time_to_us = {"ns": 1e3, "us": 1.0, "ms": 1e-3, "s": 1e-6}

    with open(path) as f:
        data = json.load(f)

    results = []
    for bench in data.get("benchmarks", []):
        m = NAME_RE.match(bench["name"])
        if not m:
            continue
        backend, dtype, dim_m, dim_n, dim_k = m.groups()
        dtype = DTYPE_LABELS.get(dtype, dtype)
        if bench.get("error_occurred"):
            print(f"Warning: skipping failed run '{bench['name']}' in {path}: "
                  f"{bench.get('error_message', 'unknown error')}", file=sys.stderr)
            continue
        flops = bench.get("FLOPS")
        if flops is None:
            print(f"Warning: no FLOPS counter for '{bench['name']}' in {path}", file=sys.stderr)
            continue
        gflops = flops / 1e9
        time_us = bench["real_time"] / time_to_us.get(bench.get("time_unit", "ns"), 1e3)
        results.append((backend, dtype, int(dim_m), int(dim_n), int(dim_k), gflops, time_us))
    return results


def main():
    parser = argparse.ArgumentParser(description="Plot bench-mul-mat-hsa results as a bar graph.")
    parser.add_argument("files", nargs="+", help="Google Benchmark JSON output files")
    parser.add_argument("--labels", help="Comma-separated labels for each file (default: filenames)")
    parser.add_argument("--metric", choices=["gflops", "time"], default="gflops",
                        help="Plot throughput (GFLOP/s, default) or wall time (ms).")
    parser.add_argument("--exclude-sizes",
                        help="Comma-separated square dims (M=N=K) to drop, e.g. '512,1024'.")
    parser.add_argument("--output", "-o", help="Save plot to file instead of showing")
    args = parser.parse_args()

    exclude_sizes = {int(s) for s in args.exclude_sizes.split(",")} if args.exclude_sizes else set()

    labels_explicit = args.labels is not None
    labels = args.labels.split(",") if args.labels else [Path(f).stem for f in args.files]
    if len(labels) != len(args.files):
        print("Error: number of labels must match number of files", file=sys.stderr)
        sys.exit(1)

    # bundles[(m, n, k, dtype)][(label, backend)] = metric value
    bundles = {}
    for path, label in zip(args.files, labels):
        results = parse_benchmark_file(path)
        if not results:
            print(f"Warning: no benchmark data found in {path}", file=sys.stderr)
            continue
        for backend, dtype, m, n, k, gflops, time_us in results:
            if m in exclude_sizes or n in exclude_sizes or k in exclude_sizes:
                continue
            key = (m, n, k, dtype)
            value = time_us / 1e3 if args.metric == "time" else gflops
            bundles.setdefault(key, {})[(label, backend)] = value

    if not bundles:
        print("Error: no benchmark data found in any file", file=sys.stderr)
        sys.exit(1)

    bundle_keys = sorted(bundles.keys())

    # Series shown per bundle: (label, backend) pairs, in first-seen order.
    # Drop the label from the series name when only one input file is given.
    series = []
    for key in bundle_keys:
        for series_key in bundles[key]:
            if series_key not in series:
                series.append(series_key)
    n_series = len(series)
    multi_file = len(args.files) > 1

    fig, ax = plt.subplots(figsize=(max(10, len(bundle_keys) * n_series * 0.6), 6))
    x = np.arange(len(bundle_keys))
    width = 0.8 / n_series

    for i, (label, backend) in enumerate(series):
        values = [bundles[key].get((label, backend), 0) for key in bundle_keys]
        offset = (i - (n_series - 1) / 2) * width
        # Explicit --labels are used verbatim as the legend text; otherwise fall
        # back to "<label>: <backend>" (multi-file) or just the backend name.
        if labels_explicit:
            bar_label = label
        else:
            bar_label = f"{label}: {backend}" if multi_file else backend
        bars = ax.bar(x + offset, values, width, label=bar_label)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                if h >= 100:
                    txt = f"{h:.0f}"
                elif h >= 1:
                    txt = f"{h:.1f}"
                else:
                    txt = f"{h:.2f}"
                ax.text(bar.get_x() + bar.get_width() / 2, h, txt,
                        ha="center", va="bottom", fontsize=16, rotation=45)

    if args.metric == "time":
        # Time spans ~4 decades across backends/sizes; log scale keeps small
        # values readable next to the large ones.
        ax.set_yscale("log")

    ax.set_xlabel("M x N x K, dtype", fontsize=16)
    ax.set_ylabel("Time (ms)" if args.metric == "time" else "Throughput (GFLOPS/s)",
                  fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}x{n}x{k}\n{dtype}" for m, n, k, dtype in bundle_keys], fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.legend(fontsize=16)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150)
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
