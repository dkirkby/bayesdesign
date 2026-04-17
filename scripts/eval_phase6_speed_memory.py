#!/usr/bin/env python3
"""Evaluate cold-start total time and peak RSS for a selected measurement mode."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from collect_phase6_cold_memory_trace import trace_case


MEASURES = {
    "full_grid": {
        "display_name": "full-grid",
        "series": (
            ("1d", "1d_full_grid", "1D sine wave", "Frequency grid points", [50, 500, 1500, 5000, 10000, 20000], "1d_full", None),
            ("3d", "3d_full_grid", "3D sine wave", "Points per parameter axis", [5, 10, 20, 30, 50], "3d_full", None),
        ),
        "default_csv": "benchmark_results/tables/macbook_full_grid.csv",
    },
    "subgrid_100": {
        "display_name": "subgrid 100 MiB",
        "series": (
            ("1d", "1d_subgrid_mem_100", "1D sine wave (subgrid 100 MiB)", "Frequency grid points", [50, 500, 1500, 5000, 10000, 20000], "1d_subgrid", 100.0),
            ("3d", "3d_subgrid_mem_100", "3D sine wave (subgrid 100 MiB)", "Points per parameter axis", [5, 10, 20, 30, 50, 100], "3d_subgrid", 100.0),
        ),
        "default_csv": "benchmark_results/tables/macbook_subgrid_100.csv",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", choices=tuple(MEASURES), required=True)
    parser.add_argument("--device-name", default="macbook_results")
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--sample-interval", type=float, default=0.005)
    parser.add_argument("--design-points-1d", type=int, default=51)
    parser.add_argument("--gpu", action="store_true", help="Run JAX on GPU and omit NumPy rows.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = MEASURES[args.measure]
    output_csv = args.output_csv or Path(config["default_csv"])

    rows = []
    for curve_family, series_key, series_label, x_label, xs, scenario, mem in config["series"]:
        mem_value = 50.0 if mem is None else mem
        for x in xs:
            backends = [("jax", args.gpu)]
            if not args.gpu:
                backends.insert(0, ("numpy", False))
            for backend_name, use_gpu in backends:
                try:
                    trace = trace_case(
                        backend_name,
                        scenario,
                        x,
                        series_label,
                        args.sample_interval,
                        "process",
                        args.design_points_1d,
                        mem_value,
                        gpu=use_gpu,
                    )
                except RuntimeError as exc:
                    print(f"Skipping {series_key} x={x} backend={backend_name}: {exc}")
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
                        "measure": args.measure,
                        "curve_family": curve_family,
                        "series_key": series_key,
                        "series_label": series_label,
                        "x_label": x_label,
                        "x_value": x,
                        "backend_key": backend_key,
                        "backend_label": backend_label,
                        "device_kind": trace["device_kind"],
                        "total_elapsed_s": trace["process_elapsed_s"],
                        "call_elapsed_s": trace["call_elapsed_s"],
                        "peak_rss_mb": trace["peak_rss_mb"],
                        "peak_gpu_mb": trace.get("peak_gpu_mb", 0.0),
                        "design_points_1d": args.design_points_1d,
                        "subgrid_mem_mb": mem_value,
                    }
                )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(output_csv)


if __name__ == "__main__":
    main()
