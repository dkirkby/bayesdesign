#!/usr/bin/env python3
"""Collect cold-start process RSS traces for one or more calculateEIG runs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
MIB = float(1 << 20)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        help="Trace case in the form backend:scenario:size:label. Repeat for multiple traces.",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--sample-interval", type=float, default=0.01)
    parser.add_argument("--design-points-1d", type=int, default=51)
    parser.add_argument("--subgrid-mem-mb", type=float, default=50.0)
    parser.add_argument("--gpu", action="store_true", help="Run JAX traces on GPU.")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--backend", choices=("numpy", "jax"))
    parser.add_argument(
        "--scenario",
        choices=("1d_full", "1d_subgrid", "3d_full", "3d_subgrid"),
    )
    parser.add_argument("--size", type=int)
    return parser.parse_args()


def parse_case(spec: str) -> tuple[str, str, int, str]:
    parts = spec.split(":", 3)
    if len(parts) != 4:
        raise ValueError(f"Invalid case spec: {spec!r}")
    backend, scenario, size, label = parts
    return backend, scenario, int(size), label


def load_numpy_backend():
    from bed.design import ExperimentDesigner
    from bed.grid import Grid, TopHat

    return {
        "name": "numpy",
        "xp": np,
        "Grid": Grid,
        "TopHat": TopHat,
        "ExperimentDesigner": ExperimentDesigner,
    }


def load_jax_backend(gpu: bool = False):
    if gpu:
        os.environ["JAX_PLATFORM_NAME"] = "gpu"
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from bed_jax.design import ExperimentDesigner
    from bed_jax.grid import Grid, TopHat

    return {
        "name": "jax",
        "xp": jnp,
        "Grid": Grid,
        "TopHat": TopHat,
        "ExperimentDesigner": ExperimentDesigner,
        "jax": jax,
    }


def _block_if_jax(backend, value):
    if backend["name"] == "jax":
        backend["jax"].block_until_ready(value)


def _sine_lfunc(params, features, designs, xp, sigma_y):
    y_mean = params.amplitude * xp.sin(params.frequency * (designs.t_obs - params.offset))
    y_diff = features.y_obs - y_mean
    return xp.exp(-0.5 * (y_diff / sigma_y) ** 2)


def build_1d_case(backend, n_param, mem=None, design_points=51):
    xp = backend["xp"]
    Grid = backend["Grid"]
    ExperimentDesigner = backend["ExperimentDesigner"]

    designs = Grid(t_obs=xp.linspace(0, 5, design_points))
    features = Grid(y_obs=xp.linspace(-1.25, 1.25, 100))
    params = Grid(
        amplitude=xp.asarray(1.0),
        frequency=xp.linspace(0.2, 2.0, n_param),
        offset=xp.asarray(0.0),
    )
    designer = ExperimentDesigner(
        params,
        features,
        designs,
        lambda p, f, d, **kwargs: _sine_lfunc(p, f, d, xp, kwargs["sigma_y"]),
        lfunc_args={"sigma_y": 0.1},
        mem=mem,
    )
    prior = params.normalize(xp.ones(params.shape))
    return designer, prior


def build_3d_case(backend, n_per_axis, mem=None):
    xp = backend["xp"]
    Grid = backend["Grid"]
    TopHat = backend["TopHat"]
    ExperimentDesigner = backend["ExperimentDesigner"]

    designs = Grid(t_obs=xp.linspace(0, 4, 32))
    features = Grid(y_obs=xp.linspace(-1.4, 1.4, 40))
    params = Grid(
        amplitude=xp.linspace(0.5, 1.5, n_per_axis),
        frequency=xp.linspace(0.2, 2.0, n_per_axis),
        offset=xp.linspace(-0.5, 0.5, n_per_axis),
    )

    def unnorm_lfunc(params, features, designs, **kwargs):
        y_mean = params.amplitude * xp.sin(params.frequency * (designs.t_obs - params.offset))
        y_diff = features.y_obs - y_mean
        return xp.exp(-0.5 * (y_diff / kwargs["sigma_y"]) ** 2)

    designer = ExperimentDesigner(
        params,
        features,
        designs,
        unnorm_lfunc,
        lfunc_args={"sigma_y": 0.1},
        mem=mem,
    )
    prior_amp = TopHat(xp.linspace(0.5, 1.5, n_per_axis))
    prior_freq = TopHat(xp.linspace(0.2, 2.0, n_per_axis))
    prior_off = TopHat(xp.linspace(-0.5, 0.5, n_per_axis))
    prior = (
        xp.asarray(prior_amp).reshape(-1, 1, 1)
        * xp.asarray(prior_freq).reshape(1, -1, 1)
        * xp.asarray(prior_off).reshape(1, 1, -1)
    )
    return designer, prior


def run_worker(args):
    backend = (
        load_numpy_backend()
        if args.backend == "numpy"
        else load_jax_backend(gpu=args.gpu)
    )
    if args.scenario == "1d_full":
        designer, prior = build_1d_case(
            backend, args.size, mem=None, design_points=args.design_points_1d
        )
    elif args.scenario == "1d_subgrid":
        designer, prior = build_1d_case(
            backend,
            args.size,
            mem=args.subgrid_mem_mb,
            design_points=args.design_points_1d,
        )
    elif args.scenario == "3d_full":
        designer, prior = build_3d_case(backend, args.size, mem=None)
    else:
        designer, prior = build_3d_case(backend, args.size, mem=args.subgrid_mem_mb)

    t0 = time.perf_counter()
    designer.calculateEIG(prior)
    _block_if_jax(backend, designer.EIG)
    elapsed = time.perf_counter() - t0
    print(json.dumps({"status": "done", "call_elapsed_s": elapsed}), flush=True)


def trace_case(
    backend: str,
    scenario: str,
    size: int,
    label: str,
    sample_interval: float,
    design_points_1d: int,
    subgrid_mem_mb: float,
    gpu: bool = False,
) -> dict:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--backend",
        backend,
        "--scenario",
        scenario,
        "--size",
        str(size),
        "--design-points-1d",
        str(design_points_1d),
        "--subgrid-mem-mb",
        str(subgrid_mem_mb),
    ]
    if gpu:
        cmd.append("--gpu")
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    ps_proc = psutil.Process(proc.pid)
    t0 = time.perf_counter()
    t_s = []
    rss_mb = []

    while proc.poll() is None:
        try:
            rss = ps_proc.memory_info().rss / MIB
        except psutil.Error:
            rss = rss_mb[-1] if rss_mb else 0.0
        t_s.append(time.perf_counter() - t0)
        rss_mb.append(rss)
        time.sleep(sample_interval)

    try:
        rss = ps_proc.memory_info().rss / MIB
    except psutil.Error:
        rss = rss_mb[-1] if rss_mb else 0.0
    t_s.append(time.perf_counter() - t0)
    rss_mb.append(rss)

    stdout = proc.stdout.read() if proc.stdout else ""
    stderr = proc.stderr.read() if proc.stderr else ""
    if proc.wait() != 0:
        raise RuntimeError(stderr or f"{backend}:{scenario} exited non-zero")

    payload = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
    if payload is None or payload.get("status") != "done":
        raise RuntimeError(f"{backend}:{scenario} did not emit completion payload")

    return {
        "backend": backend,
        "scenario": scenario,
        "size": size,
        "label": label,
        "call_elapsed_s": payload["call_elapsed_s"],
        "process_elapsed_s": t_s[-1] if t_s else 0.0,
        "peak_rss_mb": max(rss_mb) if rss_mb else 0.0,
        "device_kind": "gpu" if gpu and backend == "jax" else ("cpu" if backend == "jax" else "host"),
        "samples": {
            "t_s": t_s,
            "rss_mb": rss_mb,
        },
    }


def main():
    args = parse_args()
    if args.worker:
        run_worker(args)
        return
    if not args.case or args.output_json is None:
        raise SystemExit("--case and --output-json are required unless --worker is used")

    traces = []
    for spec in args.case:
        backend, scenario, size, label = parse_case(spec)
        traces.append(
            trace_case(
                backend,
                scenario,
                size,
                label,
                args.sample_interval,
                args.design_points_1d,
                args.subgrid_mem_mb,
                args.gpu,
            )
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(traces, indent=2), encoding="utf-8")
    print(args.output_json)


if __name__ == "__main__":
    main()
