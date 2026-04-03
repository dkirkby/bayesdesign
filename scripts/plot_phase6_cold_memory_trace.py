#!/usr/bin/env python3
"""Plot cold-start process RSS traces from a JSON array."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-codex")))

import matplotlib.pyplot as plt

DEVICE_COLOR_FAMILIES = [
    {"numpy": "#9ecae1", "jax": "tab:blue"},
    {"numpy": "#a1d99b", "jax": "tab:green"},
    {"numpy": "#fdd0a2", "jax": "tab:orange"},
    {"numpy": "#dadaeb", "jax": "tab:purple"},
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_jsons", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument(
        "--x-origin",
        choices=("process", "ready"),
        default="process",
        help="Whether to plot time from process start or shift the x-axis so worker ready is t=0.",
    )
    parser.add_argument(
        "--y-origin",
        choices=("process", "ready"),
        default="process",
        help="Whether to plot memory relative to the first process sample or the worker ready-state sample.",
    )
    parser.add_argument(
        "--metric",
        choices=("rss", "uss"),
        default="rss",
        help="Memory metric to plot. Falls back to RSS if USS samples are unavailable.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    linestyle_cycle = ["-", "--", "-.", ":"]
    linestyle_by_file = {
        path: linestyle_cycle[idx % len(linestyle_cycle)]
        for idx, path in enumerate(args.input_jsons)
    }
    color_family_by_file = {
        path: DEVICE_COLOR_FAMILIES[idx % len(DEVICE_COLOR_FAMILIES)]
        for idx, path in enumerate(args.input_jsons)
    }
    traces_by_file = {
        path: json.loads(path.read_text())
        for path in args.input_jsons
    }

    fig, ax = plt.subplots(figsize=(11, 5.5))
    backend_handles = {}
    y_label = "Worker RSS (MB)"
    synthetic_handles = []
    all_synthetic = True
    for path, traces in traces_by_file.items():
        linestyle = linestyle_by_file[path]
        for trace in traces:
            if str(trace["scenario"]).startswith("1d"):
                curve_family = "1d"
            else:
                curve_family = "3d"
            if trace.get("backend") != "synthetic":
                all_synthetic = False
            backend_key = trace.get("backend")
            if backend_key == "jax" and trace.get("device_kind") == "gpu":
                backend_key = "jax_gpu"
            elif backend_key == "jax":
                backend_key = "jax_cpu"
            family = color_family_by_file[path]
            if backend_key == "numpy":
                color = family["numpy"]
            elif backend_key in ("jax_cpu", "jax_gpu"):
                color = family["jax"]
            else:
                color = "tab:gray"
            backend_label = {
                "numpy": "NumPy",
                "jax_cpu": "JAX",
                "jax_gpu": "JAX",
            }.get(backend_key, trace["label"])
            backend_handles[(path, backend_key)] = (color, backend_label)
            expected = trace.get("expected_schedule")
            sample_key = "uss_mb" if args.metric == "uss" and "uss_mb" in trace["samples"] else "rss_mb"
            metric_name = "USS" if sample_key == "uss_mb" else "RSS"
            ready_elapsed = trace.get("ready_elapsed_s")
            if ready_elapsed is None and trace.get("expected_offset_s") is not None:
                ready_elapsed = float(trace["expected_offset_s"])
            if args.y_origin == "ready" and ready_elapsed is not None and trace["samples"]["t_s"]:
                ready_idx = min(
                    range(len(trace["samples"]["t_s"])),
                    key=lambda i: abs(trace["samples"]["t_s"][i] - ready_elapsed),
                )
                base_mem = trace["samples"][sample_key][ready_idx]
                y_values = [mem - base_mem for mem in trace["samples"][sample_key]]
                y_label = f"Worker {metric_name} delta from ready state (MB)"
            elif expected or args.y_origin == "process":
                base_mem = trace["samples"][sample_key][0] if trace["samples"][sample_key] else 0.0
                y_values = [mem - base_mem for mem in trace["samples"][sample_key]]
                y_label = f"Worker {metric_name} delta from process start (MB)"
            else:
                y_values = trace["samples"][sample_key]
                y_label = f"Worker {metric_name} (MB)"
            if args.x_origin == "ready" and ready_elapsed is not None:
                x_values = [t - ready_elapsed for t in trace["samples"]["t_s"]]
            else:
                x_values = trace["samples"]["t_s"]
            measured_line, = ax.plot(
                x_values,
                y_values,
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                label=f"{path.stem}: {trace['label']} (call={trace['call_elapsed_s']:.3f}s)",
            )

            if expected:
                stage_times = expected.get("stage_times_s", [])
                stage_sizes = expected.get("stage_sizes_mib", [])
                stage_offset = float(trace.get("expected_offset_s", 0.0))
                if stage_times and stage_sizes:
                    stair_x = [0.0]
                    stair_y = [0.0]
                    for stage_time, stage_size in zip(stage_times, stage_sizes):
                        stage_time = stage_time + stage_offset
                        if args.x_origin == "ready" and ready_elapsed is not None:
                            stage_time = stage_time - ready_elapsed
                        stair_x.extend([stage_time, stage_time])
                        stair_y.extend([stair_y[-1], stage_size])
                    end_time = x_values[-1] if x_values else stage_times[-1]
                    stair_x.append(end_time)
                    stair_y.append(stair_y[-1])
                    stair_line, = ax.plot(
                        stair_x,
                        stair_y,
                        color=color,
                        linestyle=":",
                        linewidth=1.8,
                        alpha=0.9,
                    )
                    if trace.get("backend") == "synthetic":
                        synthetic_handles = [
                            (measured_line, "Observed RSS"),
                            (stair_line, "Program-created payload"),
                        ]

    ax.set_title(args.title)
    if args.x_origin == "ready":
        ax.set_xlabel("Elapsed time relative to worker ready state (s)")
        ax.axvline(0.0, color="0.55", linestyle="--", linewidth=1.2, zorder=0)
    else:
        ax.set_xlabel("Elapsed time since worker process start (s)")
    ax.set_ylabel(y_label)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    from matplotlib.lines import Line2D

    if all_synthetic and synthetic_handles:
        ax.legend(
            handles=[handle for handle, _ in synthetic_handles],
            labels=[label for _, label in synthetic_handles],
            frameon=False,
            loc="upper left",
        )
    else:
        backend_legend = []
        seen_backend_labels = set()
        for path in args.input_jsons:
            family = color_family_by_file[path]
            for key, color in (("numpy", family["numpy"]), ("jax", family["jax"])):
                label = f"{path.stem}: {'NumPy' if key == 'numpy' else 'JAX'}"
                if label in seen_backend_labels:
                    continue
                seen_backend_labels.add(label)
                backend_legend.append(
                    Line2D([0], [0], color=color, linewidth=2.0, label=label)
                )
        device_legend = [
            Line2D([0], [0], color="0.2", linestyle=linestyle_by_file[path], linewidth=2.0, label=path.stem)
            for path in args.input_jsons
        ]
        legend1 = ax.legend(handles=backend_legend, frameon=False, loc="upper left", title="Series")
        ax.add_artist(legend1)
        if len(device_legend) > 1:
            ax.legend(handles=device_legend, frameon=False, loc="upper right", title="Device")

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
