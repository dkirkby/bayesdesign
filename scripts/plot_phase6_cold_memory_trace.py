#!/usr/bin/env python3
"""Plot cold-start process RSS traces from a JSON array."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-codex")))

import matplotlib.pyplot as plt

COLOR_BY_CURVE_BACKEND = {
    ("1d", "numpy"): "#9ecae1",
    ("1d", "jax_cpu"): "tab:blue",
    ("1d", "jax_gpu"): "tab:blue",
    ("3d", "numpy"): "#a1d99b",
    ("3d", "jax_cpu"): "tab:green",
    ("3d", "jax_gpu"): "tab:green",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_jsons", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    linestyle_cycle = ["-", "--", "-.", ":"]
    linestyle_by_file = {
        path: linestyle_cycle[idx % len(linestyle_cycle)]
        for idx, path in enumerate(args.input_jsons)
    }
    traces_by_file = {
        path: json.loads(path.read_text())
        for path in args.input_jsons
    }

    fig, ax = plt.subplots(figsize=(11, 5.5))
    backend_handles = {}
    for path, traces in traces_by_file.items():
        linestyle = linestyle_by_file[path]
        for trace in traces:
            if str(trace["scenario"]).startswith("1d"):
                curve_family = "1d"
            else:
                curve_family = "3d"
            backend_key = trace.get("backend")
            if backend_key == "jax" and trace.get("device_kind") == "gpu":
                backend_key = "jax_gpu"
            elif backend_key == "jax":
                backend_key = "jax_cpu"
            color = COLOR_BY_CURVE_BACKEND.get((curve_family, backend_key), "tab:gray")
            backend_label = {
                ("1d", "numpy"): "1D sine wave: NumPy",
                ("1d", "jax_cpu"): "1D sine wave: JAX",
                ("1d", "jax_gpu"): "1D sine wave: JAX",
                ("3d", "numpy"): "3D sine wave: NumPy",
                ("3d", "jax_cpu"): "3D sine wave: JAX",
                ("3d", "jax_gpu"): "3D sine wave: JAX",
            }.get((curve_family, backend_key), trace["label"])
            backend_handles[(curve_family, backend_key)] = (color, backend_label)
            ax.plot(
                trace["samples"]["t_s"],
                trace["samples"]["rss_mb"],
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                label=f"{path.stem}: {trace['label']} (call={trace['call_elapsed_s']:.3f}s)",
            )

    ax.set_title(args.title)
    ax.set_xlabel("Elapsed time since worker process start (s)")
    ax.set_ylabel("Worker RSS (MB)")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    from matplotlib.lines import Line2D

    backend_legend = [
        Line2D([0], [0], color=color, linewidth=2.0, label=label)
        for _, (color, label) in sorted(backend_handles.items())
    ]
    device_legend = [
        Line2D([0], [0], color="0.2", linestyle=linestyle_by_file[path], linewidth=2.0, label=path.stem)
        for path in args.input_jsons
    ]
    legend1 = ax.legend(handles=backend_legend, frameon=False, loc="upper left", title="Backend")
    ax.add_artist(legend1)
    if len(device_legend) > 1:
        ax.legend(handles=device_legend, frameon=False, loc="upper right", title="Device")

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
