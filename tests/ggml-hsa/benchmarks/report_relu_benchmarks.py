#!/usr/bin/env python3
"""Turn a bench-relu-hsa Google Benchmark JSON file into a markdown report.

Reads the JSON produced by

    ./bench-relu-hsa --benchmark_format=json --benchmark_out=results.json \
        --benchmark_repetitions=N

and emits a markdown report with a per-(dtype, shape) table of wall-clock time,
element throughput, and memory bandwidth. When the run used
--benchmark_repetitions, the mean/stddev aggregates are used; otherwise the
single-run numbers are reported.

Usage:
    ./report_relu_benchmarks.py results.json                 # print to stdout
    ./report_relu_benchmarks.py results.json -o results.md   # write to file
    ./report_relu_benchmarks.py results.json --title "NPU (HSA)"
"""

import argparse
import json
import re
import sys
from pathlib import Path

# matches individual-iteration rows (".../real_time") and aggregate rows
# (".../real_time_mean", "_stddev", "_cv", "_median"); the aggregate suffix is
# ignored here and disambiguated via the JSON run_type/aggregate_name fields.
# The four trailing integers are the ne0/ne1/ne2/ne3 shape.
NAME_RE = re.compile(
    r"^bench_relu<BackendType::(\w+),\s*(\w+)>/(\d+)/(\d+)/(\d+)/(\d+)"
    r"(?:/real_time)?(?:_\w+)?$"
)

DTYPE_LABELS = {
    "float": "F32",
}

# nanosecond conversion factors to microseconds
TIME_UNIT_TO_US = {"ns": 1e-3, "us": 1.0, "ms": 1e3, "s": 1e6}


def parse(path):
    """Parse a Google Benchmark JSON file.

    Returns (context, rows) where rows is keyed by
    (backend, dtype, ne0, ne1, ne2, ne3) -> dict with time_us_mean,
    time_us_stddev, gbps_mean, elements, iterations, repetitions, error.
    """
    with open(path) as f:
        data = json.load(f)

    context = data.get("context", {})
    rows = {}

    for bench in data.get("benchmarks", []):
        m = NAME_RE.match(bench["name"])
        if not m:
            continue
        backend, dtype, ne0, ne1, ne2, ne3 = m.groups()
        key = (
            backend,
            DTYPE_LABELS.get(dtype, dtype),
            int(ne0),
            int(ne1),
            int(ne2),
            int(ne3),
        )
        row = rows.setdefault(key, {
            "time_us_mean": None, "time_us_stddev": None,
            "gbps_mean": None, "elements": None,
            "iterations": None, "repetitions": 0, "error": None,
        })

        if bench.get("error_occurred"):
            row["error"] = bench.get("error_message", "unknown error")
            continue

        unit = TIME_UNIT_TO_US.get(bench.get("time_unit", "ns"), 1e-3)
        run_type = bench.get("run_type")
        agg = bench.get("aggregate_name")

        def gbps(b):
            # "bytes" is emitted as a per-second rate counter (bytes/s).
            v = b.get("bytes")
            return v / 1e9 if v is not None else None

        def elems(b):
            # "elements" is a per-second rate counter (elements/s); the tensor
            # size is that divided by the per-iteration rate, but for display we
            # keep the count implied by the shape (filled in below).
            return b.get("elements")

        if run_type == "aggregate" and agg == "mean":
            row["time_us_mean"] = bench["real_time"] * unit
            row["gbps_mean"] = gbps(bench)
        elif run_type == "aggregate" and agg == "stddev":
            row["time_us_stddev"] = bench["real_time"] * unit
        elif run_type == "iteration":
            row["repetitions"] += 1
            row["iterations"] = bench.get("iterations")
            if row["time_us_mean"] is None:
                row["time_us_mean"] = bench["real_time"] * unit
                row["gbps_mean"] = gbps(bench)

    # fill element counts from the shape (independent of the rate counters)
    for (backend, dt, ne0, ne1, ne2, ne3), row in rows.items():
        row["elements"] = ne0 * ne1 * ne2 * ne3

    return context, rows


def fmt(v, spec):
    return format(v, spec) if v is not None else "—"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help="Google Benchmark JSON output file")
    ap.add_argument("-o", "--output", help="Write markdown to this file (default: stdout)")
    ap.add_argument("--title", help="Report title / backend label override")
    args = ap.parse_args()

    context, rows = parse(args.file)
    if not rows:
        print(f"Error: no bench_relu data found in {args.file}", file=sys.stderr)
        sys.exit(1)

    backends = sorted({k[0] for k in rows})
    title = args.title or (backends[0] if len(backends) == 1 else "relu")

    out = []
    out.append(f"# bench-relu-hsa — {title}")
    out.append("")
    out.append(f"`out = relu(a)` (element-wise, memory-bound) via `GGML_UNARY_OP_RELU`, "
               f"from `{Path(args.file).name}`.")
    out.append("")

    # environment / context
    out.append("## Environment")
    out.append("")
    out.append("| | |")
    out.append("|---|---|")
    for label, keyname in [("Host", "host_name"), ("CPUs", "num_cpus"),
                           ("CPU MHz", "mhz_per_cpu"),
                           ("Harness build", "library_build_type"),
                           ("Timestamp", "date")]:
        if keyname in context:
            out.append(f"| {label} | {context[keyname]} |")
    if context.get("caches"):
        cache = context["caches"][-1]
        out.append(f"| Last-level cache | {cache.get('size', 0) // 1024} KiB "
                   f"(L{cache.get('level')}) |")
    out.append("")

    # results table, grouped by dtype
    out.append("## Results")
    out.append("")
    dtypes = sorted({k[1] for k in rows})
    for dtype in dtypes:
        drows = sorted(((k, v) for k, v in rows.items() if k[1] == dtype),
                       key=lambda kv: kv[1]["elements"])
        if not drows:
            continue
        out.append(f"### {dtype}")
        out.append("")
        out.append("| shape (ne0×ne1×ne2×ne3) | elements | time (µs) | stddev (µs) "
                   "| GB/s | reps × iters |")
        out.append("|---|---:|---:|---:|---:|---:|")
        for (backend, _dt, ne0, ne1, ne2, ne3), v in drows:
            shape = f"{ne0}×{ne1}×{ne2}×{ne3}"
            if v["error"]:
                out.append(f"| {shape} | {v['elements']} | *skipped* | | | {v['error']} |")
                continue
            reps = v["repetitions"] or 1
            out.append(
                f"| {shape} | {v['elements']} "
                f"| {fmt(v['time_us_mean'], '.2f')} "
                f"| {fmt(v['time_us_stddev'], '.2f')} "
                f"| {fmt(v['gbps_mean'], '.1f')} "
                f"| {reps} × {v['iterations'] or '—'} |")
        out.append("")

    text = "\n".join(out)
    if args.output:
        Path(args.output).write_text(text)
        print(f"Wrote {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
