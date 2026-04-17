#!/usr/bin/env python3
"""Evaluate and plot 3D cold-start time and peak RSS for explicit design chunk sizes."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-codex")))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from collect_phase6_cold_memory_trace import trace_case


X_VALUES = [5, 10, 20, 30, 50, 100]
NUMPY_COLOR = "#9ecae1"
JAX_COLOR = "tab:blue"
CHUNK_COLOR_BY_SIZE = {
    1: "#c6dbef",
    4: "#9ecae1",
    8: "#6baed6",
    12: "#3182bd",
    16: "#08519c",
}
LINESTYLE_BY_CHUNK = {
    16: "-",
    12: (0, (5, 1)),
    8: "--",
    4: "-.",
    1: ":",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-name", default="macbook_results")
    parser.add_argument(
        "--chunk-sizes",
        type=int,
        nargs="+",
        default=[16, 12, 8, 4, 1],
        help="Explicit design chunk sizes to compare.",
    )
    parser.add_argument("--sample-interval", type=float, default=0.005)
    parser.add_argument(
        "--plot-mode",
        choices=("absolute", "ratio"),
        default="absolute",
        help="Plot absolute quantities or NumPy/JAX ratios from the collected rows.",
    )
    parser.add_argument("--output-csv", type=Path, default=Path("benchmark_results/tables/macbook_3d_chunk_sizes.csv"))
    parser.add_argument("--output-plot", type=Path, default=Path("benchmark_results/3d_chunk_size_speed_memory_macbook_blue.png"))
    parser.add_argument("--gpu", action="store_true", help="Run JAX on GPU and omit NumPy rows.")
    parser.add_argument(
        "--metric",
        choices=("rss", "gpu"),
        default="rss",
        help="Memory metric to plot: 'rss' for peak RSS, 'gpu' for peak GPU memory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    rows = []
    for chunk_size in args.chunk_sizes:
        for x in X_VALUES:
            backends = [("jax", args.gpu)]
            if not args.gpu:
                backends.insert(0, ("numpy", False))
            for backend_name, use_gpu in backends:
                try:
                    trace = trace_case(
                        backend_name,
                        "3d_subgrid",
                        x,
                        f"3D sine wave (chunk {chunk_size})",
                        args.sample_interval,
                        "process",
                        51,
                        50.0,
                        design_chunk_size=chunk_size,
                        gpu=use_gpu,
                    )
                except RuntimeError as exc:
                    print(
                        f"Skipping chunk_size={chunk_size} x={x} backend={backend_name}: {exc}"
                    )
                    continue
                if backend_name == "numpy":
                    backend_key = "numpy"
                    backend_label = "NumPy"
                elif use_gpu:
                    backend_key = "jax_gpu"
                    backend_label = "JAX GPU"
                else:
                    backend_key = "jax_cpu"
                    backend_label = "JAX CPU"
                rows.append(
                    {
                        "table_name": args.device_name,
                        "curve_family": "3d",
                        "series_key": f"3d_chunk_{chunk_size}",
                        "series_label": f"3D sine wave (chunk {chunk_size})",
                        "x_label": "Points per parameter axis",
                        "x_value": x,
                        "backend_key": backend_key,
                        "backend_label": backend_label,
                        "device_kind": trace["device_kind"],
                        "total_elapsed_s": trace["process_elapsed_s"],
                        "call_elapsed_s": trace["call_elapsed_s"],
                        "peak_rss_mb": trace["peak_rss_mb"],
                        "peak_gpu_mb": trace.get("peak_gpu_mb", 0.0),
                        "design_chunk_size": chunk_size,
                    }
                )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(2, 1, figsize=(7.8, 7.4), sharex=True)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    mem_col = "peak_gpu_mb" if args.metric == "gpu" else "peak_rss_mb"
    mem_label = "Peak GPU Memory (MB)" if args.metric == "gpu" else "Peak RSS (MB)"

    if args.plot_mode == "absolute":
        backend_keys = [("jax_gpu", JAX_COLOR)] if args.gpu else [("numpy", NUMPY_COLOR), ("jax_cpu", JAX_COLOR)]
        for chunk_size in args.chunk_sizes:
            chunk_rows = [r for r in rows if r["design_chunk_size"] == chunk_size]
            for backend_key, color in backend_keys:
                series = sorted(
                    [r for r in chunk_rows if r["backend_key"] == backend_key],
                    key=lambda r: r["x_value"],
                )
                xs = [r["x_value"] for r in series]
                total_ys = [r["total_elapsed_s"] for r in series]
                peak_ys = [r[mem_col] for r in series]
                linestyle = LINESTYLE_BY_CHUNK.get(chunk_size, "-")
                axes[0].plot(xs, total_ys, color=color, linestyle=linestyle, marker="o", linewidth=2.0)
                axes[1].plot(xs, peak_ys, color=color, linestyle=linestyle, marker="o", linewidth=2.0)

        axes[0].set_title("3D sine wave: explicit design chunk size")
        axes[0].set_ylabel("Cold-start total time (s)")
        axes[1].set_ylabel(mem_label)

        backend_handles = [
            Line2D([0], [0], color=NUMPY_COLOR, linewidth=2.0, label="NumPy"),
            Line2D([0], [0], color=JAX_COLOR, linewidth=2.0, label="JAX"),
        ]
        axes[0].legend(handles=backend_handles, frameon=False, loc="upper left", title="Backend")
    else:
        for ax in axes:
            ax.axhline(1.0, color="0.6", linewidth=1.0, linestyle="--")
        for chunk_size in args.chunk_sizes:
            chunk_rows = [r for r in rows if r["design_chunk_size"] == chunk_size]
            points = []
            for x in sorted({r["x_value"] for r in chunk_rows}):
                numpy_rows = [r for r in chunk_rows if r["x_value"] == x and r["backend_key"] == "numpy"]
                jax_rows = [r for r in chunk_rows if r["x_value"] == x and r["backend_key"] != "numpy"]
                if not numpy_rows or not jax_rows:
                    continue
                np_row = numpy_rows[0]
                jax_row = jax_rows[0]
                np_mem = np_row[mem_col]
                jax_mem = jax_row[mem_col]
                points.append(
                    (
                        x,
                        np_row["total_elapsed_s"] / jax_row["total_elapsed_s"],
                        np_mem / jax_mem if jax_mem else 0.0,
                    )
                )
            if not points:
                continue
            xs = [p[0] for p in points]
            time_ratios = [p[1] for p in points]
            mem_ratios = [p[2] for p in points]
            color = CHUNK_COLOR_BY_SIZE.get(chunk_size, JAX_COLOR)
            axes[0].plot(xs, time_ratios, color=color, linestyle="-", marker="o", linewidth=2.2)
            axes[1].plot(xs, mem_ratios, color=color, linestyle="-", marker="o", linewidth=2.2)

        ratio_mem_label = "NumPy peak GPU mem / JAX peak GPU mem" if args.metric == "gpu" else "NumPy peak RSS / JAX peak RSS"
        axes[0].set_title("3D sine wave: NumPy/JAX ratios by design chunk size")
        axes[0].set_ylabel("NumPy time / JAX time")
        axes[1].set_ylabel(ratio_mem_label)

    axes[1].set_xlabel("Points per parameter axis")

    if args.plot_mode == "absolute":
        chunk_handles = [
            Line2D(
                [0],
                [0],
                color="0.2",
                linestyle=LINESTYLE_BY_CHUNK.get(chunk_size, "-"),
                linewidth=2.0,
                label=f"Chunk size {chunk_size}",
            )
            for chunk_size in args.chunk_sizes
        ]
    else:
        chunk_handles = [
            Line2D(
                [0],
                [0],
                color=CHUNK_COLOR_BY_SIZE.get(chunk_size, JAX_COLOR),
                linestyle="-",
                linewidth=2.2,
                label=f"Chunk size {chunk_size}",
            )
            for chunk_size in args.chunk_sizes
        ]
    fig.legend(handles=chunk_handles, frameon=False, loc="lower center", ncol=max(1, len(chunk_handles)), title="Design chunk size")
    fig.tight_layout(rect=(0, 0.07, 1, 1))

    args.output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_plot, dpi=220, bbox_inches="tight")

    print(args.output_csv)
    print(args.output_plot)


if __name__ == "__main__":
    main()
