#!/usr/bin/env python3
"""Collect cold-start process RSS traces for one or more calculateEIG runs."""

from __future__ import annotations

import argparse
import json
import os
import select
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import pynvml  # PyPI: nvidia-ml-py

ROOT = Path(__file__).resolve().parents[1]
MIB = float(1 << 20)


def _init_nvml() -> "pynvml.c_nvmlDevice_t":
    """Return an NVML device handle for GPU 0."""
    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetHandleByIndex(0)


def _gpu_mem_for_pid(handle: "pynvml.c_nvmlDevice_t | None", pid: int) -> float:
    """Return GPU memory (MiB) used by *pid*, or 0.0 when *handle* is None (non-GPU run)."""
    if handle is None:
        return 0.0
    for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
        if p.pid == pid:
            return (p.usedGpuMemory or 0) / MIB
    return 0.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        help="Trace case in the form backend:scenario:size:label. Repeat for multiple traces.",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--sample-interval", type=float, default=0.01)
    parser.add_argument(
        "--time-origin",
        choices=("process", "ready"),
        default="process",
        help="Whether the trace starts at process launch or after worker setup is ready.",
    )
    parser.add_argument("--design-points-1d", type=int, default=51)
    parser.add_argument("--feature-points-1d", type=int, default=100)
    parser.add_argument("--subgrid-mem-mb", type=float, default=50.0)
    parser.add_argument("--design-chunk-size", type=int)
    parser.add_argument(
        "--jax-mem-sample-interval",
        type=float,
        default=0.01,
        help="Sampling interval (s) for worker-side JAX memory stats.",
    )
    parser.add_argument("--gpu", action="store_true", help="Run JAX traces on GPU.")
    parser.add_argument(
        "--jax-preallocate",
        choices=("true", "false"),
        default=None,
        help="Override XLA_PYTHON_CLIENT_PREALLOCATE for JAX workers.",
    )
    parser.add_argument(
        "--jax-allocator",
        choices=("default", "platform"),
        default=None,
        help="Set XLA_PYTHON_CLIENT_ALLOCATOR for JAX workers.",
    )
    parser.add_argument(
        "--jax-mem-fraction",
        type=float,
        default=None,
        help="Set XLA_PYTHON_CLIENT_MEM_FRACTION (0,1] for JAX workers.",
    )
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


def load_jax_backend(
    gpu: bool = False,
    jax_preallocate: str | None = None,
    jax_allocator: str | None = None,
    jax_mem_fraction: float | None = None,
):
    if gpu:
        os.environ["JAX_PLATFORM_NAME"] = "gpu"
    if jax_preallocate is not None:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = jax_preallocate
    if jax_allocator is not None and jax_allocator != "default":
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = jax_allocator
    else:
        os.environ.pop("XLA_PYTHON_CLIENT_ALLOCATOR", None)
    if jax_mem_fraction is not None:
        if not (0.0 < jax_mem_fraction <= 1.0):
            raise ValueError("--jax-mem-fraction must be in (0, 1].")
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(jax_mem_fraction)
    else:
        os.environ.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
    # Enforce true cold-start behavior for benchmarking by disabling
    # persistent compilation caches in each worker process.
    os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "0"
    os.environ.pop("JAX_COMPILATION_CACHE_DIR", None)
    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_enable_compilation_cache", False)
    jax.clear_caches()
    import jax.numpy as jnp

    from bed_jax.design import ExperimentDesigner
    from bed_jax.grid import Grid, TopHat

    return {
        "name": "jax",
        "device": "gpu" if gpu else "cpu",
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


def build_1d_case(
    backend,
    n_param,
    mem=None,
    design_points=51,
    feature_points=100,
    design_chunk_size=None,
):
    xp = backend["xp"]
    Grid = backend["Grid"]
    ExperimentDesigner = backend["ExperimentDesigner"]
    device = backend.get("device")
    device_kw = {"device": device} if device is not None else {}

    designs = Grid(t_obs=xp.linspace(0, 5, design_points), **device_kw)
    features = Grid(y_obs=xp.linspace(-1.25, 1.25, feature_points), **device_kw)
    params = Grid(
        amplitude=xp.asarray(1.0),
        frequency=xp.linspace(0.2, 2.0, n_param),
        offset=xp.asarray(0.0),
        **device_kw,
    )
    designer = ExperimentDesigner(
        params,
        features,
        designs,
        lambda p, f, d, **kwargs: _sine_lfunc(p, f, d, xp, kwargs["sigma_y"]),
        lfunc_args={"sigma_y": 0.1},
        mem=mem,
        design_chunk_size=design_chunk_size,
        **device_kw,
    )
    prior = params.normalize(xp.ones(params.shape))
    return designer, prior


def build_3d_case(backend, n_per_axis, mem=None, design_chunk_size=None):
    xp = backend["xp"]
    Grid = backend["Grid"]
    TopHat = backend["TopHat"]
    ExperimentDesigner = backend["ExperimentDesigner"]
    device = backend.get("device")
    device_kw = {"device": device} if device is not None else {}

    designs = Grid(t_obs=xp.linspace(0, 4, 32), **device_kw)
    features = Grid(y_obs=xp.linspace(-1.4, 1.4, 40), **device_kw)
    params = Grid(
        amplitude=xp.linspace(0.5, 1.5, n_per_axis),
        frequency=xp.linspace(0.2, 2.0, n_per_axis),
        offset=xp.linspace(-0.5, 0.5, n_per_axis),
        **device_kw,
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
        design_chunk_size=design_chunk_size,
        **device_kw,
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
        else load_jax_backend(
            gpu=args.gpu,
            jax_preallocate=args.jax_preallocate,
            jax_allocator=args.jax_allocator,
            jax_mem_fraction=args.jax_mem_fraction,
        )
    )
    if args.scenario == "1d_full":
        designer, prior = build_1d_case(
            backend,
            args.size,
            mem=None,
            design_points=args.design_points_1d,
            feature_points=args.feature_points_1d,
            design_chunk_size=args.design_chunk_size,
        )
    elif args.scenario == "1d_subgrid":
        subgrid_mem_mb = None if args.design_chunk_size is not None else args.subgrid_mem_mb
        designer, prior = build_1d_case(
            backend,
            args.size,
            mem=subgrid_mem_mb,
            design_points=args.design_points_1d,
            feature_points=args.feature_points_1d,
            design_chunk_size=args.design_chunk_size,
        )
    elif args.scenario == "3d_full":
        designer, prior = build_3d_case(
            backend, args.size, mem=None, design_chunk_size=args.design_chunk_size
        )
    else:
        subgrid_mem_mb = None if args.design_chunk_size is not None else args.subgrid_mem_mb
        designer, prior = build_3d_case(
            backend,
            args.size,
            mem=subgrid_mem_mb,
            design_chunk_size=args.design_chunk_size,
        )

    print(json.dumps({"status": "ready"}), flush=True)
    sys.stdin.readline()

    jax_mem_samples = None
    stop_sampler = None
    sampler_thread = None
    if args.backend == "jax":
        sample_t_s = []
        sample_bytes_in_use_mb = []
        t_sample0 = time.perf_counter()
        dev = backend["jax"].devices()[0]
        stop_sampler = threading.Event()

        def _sample_jax_mem():
            while not stop_sampler.is_set():
                mem_stats = dev.memory_stats() or {}
                sample_t_s.append(time.perf_counter() - t_sample0)
                sample_bytes_in_use_mb.append(mem_stats.get("bytes_in_use", 0) / MIB)
                time.sleep(args.jax_mem_sample_interval)

        sampler_thread = threading.Thread(target=_sample_jax_mem, daemon=True)
        sampler_thread.start()

    t0 = time.perf_counter()
    designer.calculateEIG(prior)
    _block_if_jax(backend, designer.EIG)
    elapsed = time.perf_counter() - t0
    if stop_sampler is not None and sampler_thread is not None:
        stop_sampler.set()
        sampler_thread.join(timeout=1.0)
        mem_stats = backend["jax"].devices()[0].memory_stats() or {}
        sample_t_s.append(time.perf_counter() - t_sample0)
        sample_bytes_in_use_mb.append(mem_stats.get("bytes_in_use", 0) / MIB)
        jax_mem_samples = {
            "t_s": sample_t_s,
            "bytes_in_use_mb": sample_bytes_in_use_mb,
        }
    if args.backend == "numpy":
        actual_device_kind = "host"
        jax_peak_gpu_mb = 0.0
        jax_current_gpu_mb = 0.0
    else:
        dev = backend["jax"].devices()[0]
        actual_device_kind = dev.platform
        mem_stats = dev.memory_stats() or {}
        jax_peak_gpu_mb = mem_stats.get("peak_bytes_in_use", 0) / MIB
        jax_current_gpu_mb = mem_stats.get("bytes_in_use", 0) / MIB
    print(
        json.dumps(
            {
                "status": "done",
                "call_elapsed_s": elapsed,
                "actual_device_kind": actual_device_kind,
                "jax_peak_gpu_mb": jax_peak_gpu_mb,
                "jax_current_gpu_mb": jax_current_gpu_mb,
                "jax_mem_samples": jax_mem_samples,
                "jax_allocator_config": {
                    "preallocate": args.jax_preallocate,
                    "allocator": args.jax_allocator,
                    "mem_fraction": args.jax_mem_fraction,
                },
            }
        ),
        flush=True,
    )


def trace_case(
    backend: str,
    scenario: str,
    size: int,
    label: str,
    sample_interval: float,
    time_origin: str,
    design_points_1d: int,
    subgrid_mem_mb: float,
    feature_points_1d: int = 100,
    design_chunk_size: int | None = None,
    gpu: bool = False,
    jax_preallocate: str | None = None,
    jax_allocator: str | None = None,
    jax_mem_fraction: float | None = None,
) -> dict:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--time-origin",
        time_origin,
        "--backend",
        backend,
        "--scenario",
        scenario,
        "--size",
        str(size),
        "--design-points-1d",
        str(design_points_1d),
        "--feature-points-1d",
        str(feature_points_1d),
        "--subgrid-mem-mb",
        str(subgrid_mem_mb),
    ]
    if design_chunk_size is not None:
        cmd.extend(["--design-chunk-size", str(design_chunk_size)])
    if gpu:
        cmd.append("--gpu")
    if jax_preallocate is not None:
        cmd.extend(["--jax-preallocate", jax_preallocate])
    if jax_allocator is not None:
        cmd.extend(["--jax-allocator", jax_allocator])
    if jax_mem_fraction is not None:
        cmd.extend(["--jax-mem-fraction", str(jax_mem_fraction)])
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    ps_proc = psutil.Process(proc.pid)
    gpu_handle = _init_nvml() if gpu else None
    t0_process = time.perf_counter()
    ready_payload = None
    t_ready = None
    pre_t_s = []
    pre_rss_mb = []
    pre_gpu_mb = []

    while True:
        try:
            rss = ps_proc.memory_info().rss / MIB
        except psutil.Error:
            rss = pre_rss_mb[-1] if pre_rss_mb else 0.0
        gpu_mem = _gpu_mem_for_pid(gpu_handle, proc.pid)
        pre_t_s.append(time.perf_counter() - t0_process)
        pre_rss_mb.append(rss)
        pre_gpu_mb.append(gpu_mem)

        if proc.stdout and select.select([proc.stdout], [], [], sample_interval)[0]:
            line = proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("status") == "ready":
                ready_payload = payload
                t_ready = time.perf_counter()
                break

        if proc.poll() is not None:
            break

    if ready_payload is None:
        stderr = proc.stderr.read() if proc.stderr else ""
        raise RuntimeError(stderr or f"{backend}:{scenario} did not emit ready payload")

    if time_origin == "process":
        t0 = t0_process
        t_s = pre_t_s
        rss_mb = pre_rss_mb
        gpu_mb = pre_gpu_mb
    else:
        t0 = time.perf_counter()
        t_s = [0.0]
        try:
            rss0 = ps_proc.memory_info().rss / MIB
        except psutil.Error:
            rss0 = 0.0
        gpu0 = _gpu_mem_for_pid(gpu_handle, proc.pid)
        rss_mb = [rss0]
        gpu_mb = [gpu0]

    if proc.stdin:
        proc.stdin.write("go\n")
        proc.stdin.flush()
        proc.stdin.close()

    while proc.poll() is None:
        try:
            rss = ps_proc.memory_info().rss / MIB
        except psutil.Error:
            rss = rss_mb[-1] if rss_mb else 0.0
        gpu_mem = _gpu_mem_for_pid(gpu_handle, proc.pid)
        t_s.append(time.perf_counter() - t0)
        rss_mb.append(rss)
        gpu_mb.append(gpu_mem)
        time.sleep(sample_interval)

    try:
        rss = ps_proc.memory_info().rss / MIB
    except psutil.Error:
        rss = rss_mb[-1] if rss_mb else 0.0
    gpu_mem = _gpu_mem_for_pid(gpu_handle, proc.pid)
    t_s.append(time.perf_counter() - t0)
    rss_mb.append(rss)
    gpu_mb.append(gpu_mem)

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

    jax_gpu_mb = [0.0] * len(t_s)
    jax_mem_samples = payload.get("jax_mem_samples")
    if jax_mem_samples and t_s:
        worker_t = [float(v) for v in jax_mem_samples.get("t_s", [])]
        worker_mem = [float(v) for v in jax_mem_samples.get("bytes_in_use_mb", [])]
        if worker_t and worker_mem and len(worker_t) == len(worker_mem):
            if time_origin == "process":
                ready_elapsed_s = (t_ready - t0_process) if t_ready is not None else 0.0
                mapped_t = [ready_elapsed_s + dt for dt in worker_t]
            else:
                mapped_t = worker_t
            jax_gpu_mb = np.interp(
                np.asarray(t_s, dtype=float),
                np.asarray(mapped_t, dtype=float),
                np.asarray(worker_mem, dtype=float),
                left=worker_mem[0],
                right=worker_mem[-1],
            ).tolist()

    return {
        "backend": backend,
        "scenario": scenario,
        "size": size,
        "label": label,
        "call_elapsed_s": payload["call_elapsed_s"],
        "process_elapsed_s": t_s[-1] if t_s else 0.0,
        "peak_rss_mb": max(rss_mb) if rss_mb else 0.0,
        "peak_gpu_mb": max(gpu_mb) if gpu_mb else 0.0,
        "jax_peak_gpu_mb": payload.get("jax_peak_gpu_mb", 0.0),
        "jax_allocator_config": payload.get(
            "jax_allocator_config",
            {
                "preallocate": jax_preallocate,
                "allocator": jax_allocator,
                "mem_fraction": jax_mem_fraction,
            },
        ),
        "time_origin": time_origin,
        "ready_elapsed_s": (t_ready - t0_process) if t_ready is not None else None,
        "device_kind": payload.get(
            "actual_device_kind",
            "gpu" if gpu and backend == "jax" else ("cpu" if backend == "jax" else "host"),
        ),
        "samples": {
            "t_s": t_s,
            "rss_mb": rss_mb,
            "gpu_mb": gpu_mb,
            "jax_gpu_mb": jax_gpu_mb,
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
                args.time_origin,
                args.design_points_1d,
                args.subgrid_mem_mb,
                feature_points_1d=args.feature_points_1d,
                design_chunk_size=args.design_chunk_size,
                gpu=args.gpu,
                jax_preallocate=args.jax_preallocate,
                jax_allocator=args.jax_allocator,
                jax_mem_fraction=args.jax_mem_fraction,
            )
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(traces, indent=2), encoding="utf-8")
    print(args.output_json)


if __name__ == "__main__":
    main()
