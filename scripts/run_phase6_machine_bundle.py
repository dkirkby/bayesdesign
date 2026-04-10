#!/usr/bin/env python3
"""Collect + plot 3D full/subgrid cold-memory comparisons for one machine.

This script is intended to be run once per machine (for example:
`--machine entropy --color green`). It enforces CPU execution for JAX traces,
stores machine-scoped JSON/CSV artifacts, and regenerates combined plots that
overlay all machines found in benchmark_results/machines.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-codex")))

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from collect_phase6_cold_memory_trace import trace_case


POINTS = [10, 20, 40, 80]
CHUNKS = [20, 10, 5, 1]
CHUNK_LIGHTEN = {20: 0.00, 10: 0.16, 5: 0.30, 1: 0.46}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", required=True, help="Machine label, e.g. macbook, entropy, perlmutter.")
    parser.add_argument(
        "--color",
        required=True,
        help='Base matplotlib color for this machine, e.g. "green" or "tab:orange".',
    )
    parser.add_argument("--output-root", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--sample-interval", type=float, default=0.01)
    parser.add_argument("--skip-collect", action="store_true", help="Only regenerate plots/CSVs from existing JSON.")
    return parser.parse_args()


def normalize_color(spec: str) -> str:
    s = spec.strip()
    plain = s.lower()
    if plain in {"blue", "green", "orange", "red", "purple", "brown", "pink", "gray", "olive", "cyan"}:
        return f"tab:{plain}"
    # Validate color string now.
    mcolors.to_rgb(s)
    return s


def lighten(color: str, amount: float) -> tuple[float, float, float]:
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def load_styles(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_styles(path: Path, styles: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(styles, indent=2, sort_keys=True), encoding="utf-8")


def machine_dir(root: Path, machine: str) -> Path:
    return root / "machines" / machine


def machine_full_json(root: Path, machine: str) -> Path:
    return machine_dir(root, machine) / "cold_memory_trace_3d_full_strict_cold.json"


def machine_chunk_json(root: Path, machine: str, chunk: int) -> Path:
    return machine_dir(root, machine) / f"cold_memory_trace_3d_subgrid_chunk{chunk}_strict_cold.json"


def ensure_legacy_macbook_seed(root: Path, styles: dict[str, str]) -> None:
    """If older root-level canonical files exist, seed them into machines/macbook."""
    legacy_full = root / "cold_memory_trace_3d_full_strict_cold.json"
    legacy_chunk = {c: root / f"cold_memory_trace_3d_subgrid_chunk{c}_strict_cold.json" for c in CHUNKS}
    if not legacy_full.exists() or not all(p.exists() for p in legacy_chunk.values()):
        return

    mac_dir = machine_dir(root, "macbook")
    mac_dir.mkdir(parents=True, exist_ok=True)
    full_out = machine_full_json(root, "macbook")
    if not full_out.exists():
        shutil.copy2(legacy_full, full_out)
    for c, src in legacy_chunk.items():
        dst = machine_chunk_json(root, "macbook", c)
        if not dst.exists():
            shutil.copy2(src, dst)
    styles.setdefault("macbook", "tab:blue")


def collect_machine(root: Path, machine: str, sample_interval: float) -> None:
    # Enforce CPU paths for JAX traces.
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    mdir = machine_dir(root, machine)
    mdir.mkdir(parents=True, exist_ok=True)

    full_traces = []
    for p in POINTS:
        full_traces.append(
            trace_case(
                "numpy",
                "3d_full",
                p,
                "NumPy full-grid",
                sample_interval,
                "process",
                51,
                50.0,
                design_chunk_size=None,
                gpu=False,
            )
        )
        full_traces.append(
            trace_case(
                "jax",
                "3d_full",
                p,
                "JAX full-grid",
                sample_interval,
                "process",
                51,
                50.0,
                design_chunk_size=None,
                gpu=False,
            )
        )
    full_out = machine_full_json(root, machine)
    full_out.write_text(json.dumps(full_traces, indent=2), encoding="utf-8")
    for t in full_traces:
        if t["backend"] == "jax" and t.get("device_kind") != "cpu":
            raise RuntimeError(f"Expected CPU JAX trace, got device_kind={t.get('device_kind')}")

    for c in CHUNKS:
        traces = []
        for p in POINTS:
            traces.append(
                trace_case(
                    "numpy",
                    "3d_subgrid",
                    p,
                    f"NumPy chunk{c}",
                    sample_interval,
                    "process",
                    51,
                    50.0,
                    design_chunk_size=c,
                    gpu=False,
                )
            )
            traces.append(
                trace_case(
                    "jax",
                    "3d_subgrid",
                    p,
                    f"JAX chunk{c}",
                    sample_interval,
                    "process",
                    51,
                    50.0,
                    design_chunk_size=c,
                    gpu=False,
                )
            )
        chunk_out = machine_chunk_json(root, machine, c)
        chunk_out.write_text(json.dumps(traces, indent=2), encoding="utf-8")
        for t in traces:
            if t["backend"] == "jax" and t.get("device_kind") != "cpu":
                raise RuntimeError(f"Expected CPU JAX trace, got device_kind={t.get('device_kind')}")


def load_machine_traces(root: Path, machine: str) -> dict:
    full_path = machine_full_json(root, machine)
    if not full_path.exists():
        raise FileNotFoundError(str(full_path))
    out = {
        "full": json.loads(full_path.read_text(encoding="utf-8")),
        "chunks": {},
    }
    for c in CHUNKS:
        p = machine_chunk_json(root, machine, c)
        if p.exists():
            out["chunks"][c] = json.loads(p.read_text(encoding="utf-8"))
    return out


def ratios_from_traces(traces: list[dict], scenario: str) -> dict[int, tuple[float, float]]:
    by_size: dict[int, dict[str, dict]] = {}
    for t in traces:
        if t.get("scenario") != scenario:
            continue
        by_size.setdefault(int(t["size"]), {})[t["backend"]] = t
    out: dict[int, tuple[float, float]] = {}
    for p in POINTS:
        rec = by_size.get(p, {})
        if "numpy" not in rec or "jax" not in rec:
            continue
        n = rec["numpy"]
        j = rec["jax"]
        out[p] = (
            float(n["process_elapsed_s"]) / float(j["process_elapsed_s"]),
            float(n["peak_rss_mb"]) / float(j["peak_rss_mb"]),
        )
    return out


def write_machine_ratio_csvs(root: Path, machine: str, traces: dict) -> None:
    mdir = machine_dir(root, machine)
    mdir.mkdir(parents=True, exist_ok=True)

    full_ratio = ratios_from_traces(traces["full"], "3d_full")
    with (mdir / "full_grid_3d_ratio.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["points_per_axis", "numpy_over_jax_time", "numpy_over_jax_peak_rss"])
        for p in POINTS:
            if p in full_ratio:
                t, m = full_ratio[p]
                w.writerow([p, f"{t:.10f}", f"{m:.10f}"])

    with (mdir / "subgrid_chunk_ratio.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["design_chunk_size", "points_per_axis", "numpy_over_jax_time", "numpy_over_jax_peak_rss"])
        for c in CHUNKS:
            if c not in traces["chunks"]:
                continue
            ratio = ratios_from_traces(traces["chunks"][c], "3d_subgrid")
            for p in POINTS:
                if p in ratio:
                    t, m = ratio[p]
                    w.writerow([c, p, f"{t:.10f}", f"{m:.10f}"])


def to_ready_relative(trace: dict) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(trace["samples"]["t_s"], dtype=float)
    r = np.asarray(trace["samples"]["rss_mb"], dtype=float)
    t_ready = float(trace["ready_elapsed_s"])
    r_ready = np.interp(t_ready, t, r)
    return t - t_ready, r - r_ready


def pick_trace(traces: list[dict], scenario: str, size: int, backend: str) -> dict | None:
    for t in traces:
        if t.get("scenario") == scenario and int(t["size"]) == size and t.get("backend") == backend:
            return t
    return None


def plot_full_ratio(root: Path, all_data: dict[str, dict], styles: dict[str, str]) -> Path:
    out = root / "3d_full_grid_ratio_all_machines.png"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.6, 8.8), sharex=True)
    for machine, data in sorted(all_data.items()):
        ratio = ratios_from_traces(data["full"], "3d_full")
        xs = [p for p in POINTS if p in ratio]
        if not xs:
            continue
        t = [ratio[p][0] for p in xs]
        m = [ratio[p][1] for p in xs]
        color = styles[machine]
        ax1.plot(xs, t, color=color, marker="o", linewidth=2.2, label=machine)
        ax2.plot(xs, m, color=color, marker="o", linewidth=2.2, label=machine)
    for ax in (ax1, ax2):
        ax.axhline(1.0, color="0.55", linestyle="--", linewidth=1.3)
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax1.set_title("3D sine wave: full-grid NumPy/JAX ratios")
    ax1.set_ylabel("NumPy time / JAX time")
    ax2.set_ylabel("NumPy peak RSS / JAX peak RSS")
    ax2.set_xlabel("Points per parameter axis")
    ax1.set_xlim(10, 80)
    ax2.set_xlim(10, 80)
    ax2.set_xticks(list(range(10, 81, 10)))
    ax1.legend(frameon=False, title="Machine")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    return out


def plot_subgrid_chunk_ratio(root: Path, all_data: dict[str, dict], styles: dict[str, str]) -> Path:
    out = root / "3d_subgrid_chunk_ratio_all_machines.png"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.4, 9.0), sharex=True)
    for machine, data in sorted(all_data.items()):
        base = styles[machine]
        for c in CHUNKS:
            traces = data["chunks"].get(c)
            if not traces:
                continue
            ratio = ratios_from_traces(traces, "3d_subgrid")
            xs = [p for p in POINTS if p in ratio]
            if not xs:
                continue
            t = [ratio[p][0] for p in xs]
            m = [ratio[p][1] for p in xs]
            color = lighten(base, CHUNK_LIGHTEN[c])
            label = f"{machine} chunk={c}"
            ax1.plot(xs, t, color=color, marker="o", linewidth=2.0, label=label)
            ax2.plot(xs, m, color=color, marker="o", linewidth=2.0, label=label)
    for ax in (ax1, ax2):
        ax.axhline(1.0, color="0.55", linestyle="--", linewidth=1.3)
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax1.set_title("3D sine wave: subgrid chunk-size NumPy/JAX ratios")
    ax1.set_ylabel("NumPy time / JAX time")
    ax2.set_ylabel("NumPy peak RSS / JAX peak RSS")
    ax2.set_xlabel("Points per parameter axis")
    ax1.set_xlim(10, 80)
    ax2.set_xlim(10, 80)
    ax2.set_xticks(list(range(10, 81, 10)))
    ax1.legend(frameon=False, ncol=2, fontsize=9, title="Machine + chunk")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    return out


def plot_time_series_full(root: Path, all_data: dict[str, dict], styles: dict[str, str]) -> Path:
    out = root / "cold_memory_trace_3d_full_p80_timeseries_ready_all_machines.png"
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for machine, data in sorted(all_data.items()):
        base = styles[machine]
        n = pick_trace(data["full"], "3d_full", 80, "numpy")
        j = pick_trace(data["full"], "3d_full", 80, "jax")
        if not n or not j:
            continue
        tn, rn = to_ready_relative(n)
        tj, rj = to_ready_relative(j)
        ax.plot(tn, rn, color=base, linestyle="--", linewidth=2.0, label=f"{machine} NumPy")
        ax.plot(tj, rj, color=base, linestyle="-", linewidth=2.0, label=f"{machine} JAX")
    ax.axvline(0.0, color="0.5", linestyle="--", linewidth=1.4)
    ax.set_title("3D full-grid (points=80): memory time series")
    ax.set_xlabel("Elapsed time relative to worker ready state (s)")
    ax.set_ylabel("Worker RSS delta from ready state (MB)")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    return out


def plot_time_series_chunk5(root: Path, all_data: dict[str, dict], styles: dict[str, str]) -> Path:
    out = root / "cold_memory_trace_3d_subgrid_chunk5_p80_timeseries_ready_all_machines.png"
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for machine, data in sorted(all_data.items()):
        base = styles[machine]
        color = lighten(base, CHUNK_LIGHTEN[5])
        traces = data["chunks"].get(5)
        if not traces:
            continue
        n = pick_trace(traces, "3d_subgrid", 80, "numpy")
        j = pick_trace(traces, "3d_subgrid", 80, "jax")
        if not n or not j:
            continue
        tn, rn = to_ready_relative(n)
        tj, rj = to_ready_relative(j)
        ax.plot(tn, rn, color=color, linestyle="--", linewidth=2.0, label=f"{machine} NumPy")
        ax.plot(tj, rj, color=color, linestyle="-", linewidth=2.0, label=f"{machine} JAX")
    ax.axvline(0.0, color="0.5", linestyle="--", linewidth=1.4)
    ax.set_title("3D subgrid (chunk=5, points=80): memory time series")
    ax.set_xlabel("Elapsed time relative to worker ready state (s)")
    ax.set_ylabel("Worker RSS delta from ready state (MB)")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    return out


def main() -> None:
    args = parse_args()
    root = args.output_root
    styles_path = root / "machines" / "styles.json"

    styles = load_styles(styles_path)
    ensure_legacy_macbook_seed(root, styles)
    styles[args.machine] = normalize_color(args.color)
    save_styles(styles_path, styles)

    if not args.skip_collect:
        collect_machine(root, args.machine, args.sample_interval)

    all_data: dict[str, dict] = {}
    for machine in sorted(styles):
        mdir = machine_dir(root, machine)
        if not mdir.exists():
            continue
        try:
            all_data[machine] = load_machine_traces(root, machine)
        except FileNotFoundError:
            continue

    if args.machine not in all_data:
        raise SystemExit(
            f"No machine data found for '{args.machine}'. "
            f"Run without --skip-collect or add JSONs under {machine_dir(root, args.machine)}"
        )

    for machine, traces in all_data.items():
        write_machine_ratio_csvs(root, machine, traces)

    p1 = plot_full_ratio(root, all_data, styles)
    p2 = plot_subgrid_chunk_ratio(root, all_data, styles)
    p3 = plot_time_series_full(root, all_data, styles)
    p4 = plot_time_series_chunk5(root, all_data, styles)

    print(f"styles: {styles_path}")
    print(f"machine_data: {machine_dir(root, args.machine)}")
    print(p1)
    print(p2)
    print(p3)
    print(p4)


if __name__ == "__main__":
    main()
