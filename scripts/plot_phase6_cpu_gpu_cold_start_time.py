#!/usr/bin/env python3
"""Plot JAX cold-start wall time (CPU vs GPU) for the same machine and grid sizes.

Reads the JSON artifacts produced by ``collect_phase6_cold_memory_trace`` /
``run_phase6_machine_bundle``:

- CPU: ``benchmark_results/machines/<machine>/cold_memory_trace_3d_*_strict_cold.json``
- GPU: same stem with ``_gpu`` before ``.json``

Uses ``process_elapsed_s`` (wall time from process start through the end of the
worker trace), matching the other Phase 6 cold-start plots. Only **JAX** rows
are compared (``backend == "jax"``), with ``device_kind`` ``cpu`` vs ``gpu``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-codex")))

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _normalize_color(spec: str) -> str:
    s = spec.strip()
    plain = s.lower()
    if plain in {
        "blue",
        "green",
        "orange",
        "red",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    }:
        return f"tab:{plain}"
    mcolors.to_rgb(s)
    return s


def _machine_line_color(root: Path, machine: str) -> str:
    """Match ``run_phase6_machine_bundle`` / ``styles.json`` (e.g. entropy → green)."""
    path = root / "machines" / "styles.json"
    if path.exists():
        styles = json.loads(path.read_text(encoding="utf-8"))
        if machine in styles:
            return _normalize_color(styles[machine])
    return "tab:green"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--machine", required=True, help="Machine label under benchmark_results/machines/")
    p.add_argument("--output-root", type=Path, default=Path("benchmark_results"))
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG output path (default: <output-root>/cold_start_cpu_gpu_<machine>.png).",
    )
    p.add_argument(
        "--scenario",
        choices=("3d_full", "3d_subgrid"),
        default="3d_full",
        help="Which cold JSON stem to load.",
    )
    p.add_argument(
        "--chunk",
        type=int,
        default=None,
        help="Design chunk size for 3d_subgrid (required when --scenario 3d_subgrid).",
    )
    return p.parse_args()


def _json_path(root: Path, machine: str, scenario: str, gpu: bool, chunk: int | None) -> Path:
    mdir = root / "machines" / machine
    suffix = "_gpu" if gpu else ""
    if scenario == "3d_full":
        return mdir / f"cold_memory_trace_3d_full_strict_cold{suffix}.json"
    if chunk is None:
        raise SystemExit("--chunk is required when --scenario 3d_subgrid")
    return mdir / f"cold_memory_trace_3d_subgrid_chunk{chunk}_strict_cold{suffix}.json"


def _load_times(path: Path, scenario: str, want_kind: str) -> dict[int, float]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    traces = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, float] = {}
    for t in traces:
        if t.get("scenario") != scenario or t.get("backend") != "jax":
            continue
        if t.get("device_kind") != want_kind:
            continue
        out[int(t["size"])] = float(t["process_elapsed_s"])
    return out


def main() -> None:
    args = parse_args()
    if args.scenario == "3d_subgrid" and args.chunk is None:
        raise SystemExit("--chunk is required when --scenario 3d_subgrid")

    cpu_path = _json_path(args.output_root, args.machine, args.scenario, gpu=False, chunk=args.chunk)
    gpu_path = _json_path(args.output_root, args.machine, args.scenario, gpu=True, chunk=args.chunk)

    cpu_by_n = _load_times(cpu_path, args.scenario, "cpu")
    gpu_by_n = _load_times(gpu_path, args.scenario, "gpu")

    xs = sorted(set(cpu_by_n) & set(gpu_by_n))
    if not xs:
        raise SystemExit(
            f"No overlapping JAX sizes between CPU and GPU traces for {args.machine!r}. "
            f"CPU path={cpu_path} (sizes={sorted(cpu_by_n)}), "
            f"GPU path={gpu_path} (sizes={sorted(gpu_by_n)})."
        )

    y_cpu = [cpu_by_n[n] for n in xs]
    y_gpu = [gpu_by_n[n] for n in xs]

    line_color = _machine_line_color(args.output_root, args.machine)

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot(
        xs,
        y_cpu,
        color=line_color,
        linestyle="-",
        marker="o",
        linewidth=2.2,
    )
    ax.plot(
        xs,
        y_gpu,
        color=line_color,
        linestyle="--",
        marker="s",
        linewidth=2.2,
    )
    ax.set_xlabel("Points per parameter axis")
    ax.set_ylabel("Cold-start wall time (s)")
    sub = f"{args.scenario}" + (
        "" if args.scenario == "3d_full" else f", chunk={args.chunk}"
    )
    ax.set_title(f"{args.machine}: JAX cold-start time ({sub})")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Long handles so dashed vs solid is visible in the legend.
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=line_color,
            linestyle="-",
            linewidth=2.2,
            marker="o",
            markersize=7,
            markerfacecolor=line_color,
            markeredgecolor=line_color,
            label="JAX CPU",
        ),
        Line2D(
            [0],
            [0],
            color=line_color,
            linestyle="--",
            dash_capstyle="round",
            linewidth=2.2,
            marker="s",
            markersize=6,
            markerfacecolor=line_color,
            markeredgecolor=line_color,
            label="JAX GPU",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        handlelength=4.2,
        handletextpad=0.6,
    )
    fig.tight_layout()

    out = args.output
    if out is None:
        out = args.output_root / f"cold_start_cpu_gpu_{args.machine}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
