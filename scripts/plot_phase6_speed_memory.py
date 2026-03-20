#!/usr/bin/env python3
"""Plot cold-start total time and peak RSS from one or more device CSVs."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-codex")))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


LINESTYLES = ["-", "--", "-.", ":"]
COLOR_BY_CURVE_BACKEND = {
    ("1d", "numpy"): "#9ecae1",
    ("1d", "jax_cpu"): "tab:blue",
    ("1d", "jax_gpu"): "tab:blue",
    ("3d", "numpy"): "#a1d99b",
    ("3d", "jax_cpu"): "tab:green",
    ("3d", "jax_gpu"): "tab:green",
}
LABEL_BY_CURVE_BACKEND = {
    ("1d", "numpy"): "1D sine wave: NumPy",
    ("1d", "jax_cpu"): "1D sine wave: JAX",
    ("1d", "jax_gpu"): "1D sine wave: JAX",
    ("3d", "numpy"): "3D sine wave: NumPy",
    ("3d", "jax_cpu"): "3D sine wave: JAX",
    ("3d", "jax_gpu"): "3D sine wave: JAX",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_paths", type=Path, nargs="+")
    parser.add_argument("--measure", choices=("full_grid", "subgrid_100"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["x_value"] = float(row["x_value"])
        row["total_elapsed_s"] = float(row["total_elapsed_s"])
        row["peak_rss_mb"] = float(row["peak_rss_mb"])
    return rows


def main():
    args = parse_args()
    rows_by_csv = {path: _load_rows(path) for path in args.csv_paths}
    linestyle_by_csv = {
        path: LINESTYLES[idx % len(LINESTYLES)] for idx, path in enumerate(args.csv_paths)
    }

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), sharey="row")
    ax_t1, ax_t3 = axes[0]
    ax_m1, ax_m3 = axes[1]
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    grouped = defaultdict(list)
    for path, rows in rows_by_csv.items():
        for row in rows:
            grouped[(path, row["curve_family"], row["backend_key"])].append(row)

    for (path, curve_family, backend_key), rows in grouped.items():
        rows = sorted(rows, key=lambda r: r["x_value"])
        xs = [r["x_value"] for r in rows]
        time_ys = [r["total_elapsed_s"] for r in rows]
        mem_ys = [r["peak_rss_mb"] for r in rows]
        color = COLOR_BY_CURVE_BACKEND[(curve_family, backend_key)]
        linestyle = linestyle_by_csv[path]
        if curve_family == "1d":
            ax_t1.plot(xs, time_ys, color=color, linestyle=linestyle, marker="o", linewidth=2.0)
            ax_m1.plot(xs, mem_ys, color=color, linestyle=linestyle, marker="o", linewidth=2.0)
        else:
            ax_t3.plot(xs, time_ys, color=color, linestyle=linestyle, marker="o", linewidth=2.0)
            ax_m3.plot(xs, mem_ys, color=color, linestyle=linestyle, marker="o", linewidth=2.0)

    title_suffix = "full-grid" if args.measure == "full_grid" else "subgrid 100 MiB"
    ax_t1.set_title(f"1D sine wave: {title_suffix}")
    ax_t3.set_title(f"3D sine wave: {title_suffix}")
    ax_t1.set_ylabel("Cold-start total time (s)")
    ax_m1.set_ylabel("Peak RSS (MB)")
    ax_t1.set_xlabel("Frequency grid points")
    ax_m1.set_xlabel("Frequency grid points")
    ax_t3.set_xlabel("Points per parameter axis")
    ax_m3.set_xlabel("Points per parameter axis")

    curve_handles = [
        Line2D(
            [0],
            [0],
            color=COLOR_BY_CURVE_BACKEND[key],
            linewidth=2.0,
            label=LABEL_BY_CURVE_BACKEND[key],
        )
        for key in (
            ("1d", "numpy"),
            ("1d", "jax_cpu"),
            ("3d", "numpy"),
            ("3d", "jax_cpu"),
        )
    ]
    device_handles = [
        Line2D([0], [0], color="0.2", linestyle=linestyle_by_csv[path], linewidth=2.0, label=path.stem)
        for path in args.csv_paths
    ]

    present_backend_keys = {
        (row["curve_family"], row["backend_key"])
        for rows in rows_by_csv.values()
        for row in rows
    }
    curve_handles = [
        handle
        for handle, key in zip(
            curve_handles,
            (
                ("1d", "numpy"),
                ("1d", "jax_cpu"),
                ("3d", "numpy"),
                ("3d", "jax_cpu"),
            ),
        )
        if key in present_backend_keys or (key[1] == "jax_cpu" and (key[0], "jax_gpu") in present_backend_keys)
    ]

    ax_t1.legend(
        handles=[h for h in curve_handles if h.get_label().startswith("1D")],
        frameon=False,
        loc="upper left",
        title="Backend",
    )
    ax_t3.legend(
        handles=[h for h in curve_handles if h.get_label().startswith("3D")],
        frameon=False,
        loc="upper left",
        title="Backend",
    )
    fig.legend(handles=device_handles, frameon=False, loc="lower center", ncol=max(1, len(device_handles)), title="Device")

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
